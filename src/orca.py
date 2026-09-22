"""
Pure-Python ORCA (Optimal Reciprocal Collision Avoidance).

Faithful port of the linear-programming core of the reference RVO2 C++ library
(van den Berg, Guy, Lin & Manocha, 2011) -- the same algorithm that `python-rvo2`
wraps.  Kept dependency-free so that the crowd policy is bit-for-bit reproducible
on any machine and the notebook never depends on a Cython/CMake build succeeding.

Only agent-agent half-planes are used (no static obstacle lines), which matches
the circle-crossing benchmark used by CrowdNav / CADRL / SARL.
"""
import math

EPSILON = 1e-5


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _abs_sq(v):
    return v[0] * v[0] + v[1] * v[1]


def _norm(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1])
    if n < EPSILON:
        return (0.0, 0.0)
    return (v[0] / n, v[1] / n)


def _linear_program1(lines, line_no, radius, opt_velocity, direction_opt):
    """Optimise along the boundary of line `line_no`. Returns (ok, result)."""
    lp, ld = lines[line_no]
    dot_product = lp[0] * ld[0] + lp[1] * ld[1]
    discriminant = dot_product * dot_product + radius * radius - _abs_sq(lp)

    if discriminant < 0.0:
        return False, None

    sqrt_disc = math.sqrt(discriminant)
    t_left = -dot_product - sqrt_disc
    t_right = -dot_product + sqrt_disc

    for i in range(line_no):
        ip, idir = lines[i]
        denominator = _det(ld, idir)
        numerator = _det(idir, (lp[0] - ip[0], lp[1] - ip[1]))

        if abs(denominator) <= EPSILON:
            # Lines are (near) parallel.
            if numerator < 0.0:
                return False, None
            continue

        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)

        if t_left > t_right:
            return False, None

    if direction_opt:
        t = t_right if (opt_velocity[0] * ld[0] + opt_velocity[1] * ld[1]) > 0.0 else t_left
    else:
        t = ld[0] * (opt_velocity[0] - lp[0]) + ld[1] * (opt_velocity[1] - lp[1])
        if t < t_left:
            t = t_left
        elif t > t_right:
            t = t_right

    return True, (lp[0] + t * ld[0], lp[1] + t * ld[1])


def _linear_program2(lines, radius, opt_velocity, direction_opt):
    """Returns (index_of_first_failing_line, result)."""
    if direction_opt:
        result = (opt_velocity[0] * radius, opt_velocity[1] * radius)
    elif _abs_sq(opt_velocity) > radius * radius:
        u = _norm(opt_velocity)
        result = (u[0] * radius, u[1] * radius)
    else:
        result = (opt_velocity[0], opt_velocity[1])

    for i in range(len(lines)):
        ip, idir = lines[i]
        if _det(idir, (ip[0] - result[0], ip[1] - result[1])) > 0.0:
            ok, new_result = _linear_program1(lines, i, radius, opt_velocity, direction_opt)
            if not ok:
                return i, result
            result = new_result

    return len(lines), result


def _linear_program3(lines, begin_line, radius, result):
    """Relax the ORCA constraints uniformly when the LP is infeasible (dense crowds)."""
    distance = 0.0

    for i in range(begin_line, len(lines)):
        ip, idir = lines[i]
        if _det(idir, (ip[0] - result[0], ip[1] - result[1])) > distance:
            proj_lines = []
            for j in range(i):
                jp, jdir = lines[j]
                determinant = _det(idir, jdir)

                if abs(determinant) <= EPSILON:
                    if idir[0] * jdir[0] + idir[1] * jdir[1] > 0.0:
                        continue  # same direction
                    point = (0.5 * (ip[0] + jp[0]), 0.5 * (ip[1] + jp[1]))
                else:
                    s = _det(jdir, (ip[0] - jp[0], ip[1] - jp[1])) / determinant
                    point = (ip[0] + s * idir[0], ip[1] + s * idir[1])

                direction = _norm((jdir[0] - idir[0], jdir[1] - idir[1]))
                proj_lines.append((point, direction))

            temp_result = result
            n_fail, new_result = _linear_program2(
                proj_lines, radius, (-idir[1], idir[0]), True
            )
            if n_fail < len(proj_lines):
                result = temp_result
            else:
                result = new_result

            distance = _det(idir, (ip[0] - result[0], ip[1] - result[1]))

    return result


def orca_velocity(pos, vel, radius, pref_vel, max_speed,
                  neighbours, time_horizon=5.0, time_step=0.25,
                  neighbour_dist=10.0, max_neighbours=10, responsibility=0.5):
    """Compute one agent's new ORCA velocity.

    Parameters
    ----------
    pos, vel, pref_vel : (x, y) tuples for this agent
    radius, max_speed  : scalars for this agent
    neighbours         : list of (pos, vel, radius) tuples for other agents
    responsibility     : share of the avoidance effort this agent takes.  0.5 is
                         the reciprocal (mutual) case; use 1.0 when the other
                         agents cannot see this agent and will not cooperate.
    """
    inv_time_horizon = 1.0 / time_horizon
    inv_time_step = 1.0 / time_step

    # Keep the closest `max_neighbours` within `neighbour_dist` (as RVO2 does).
    cand = []
    nd_sq = neighbour_dist * neighbour_dist
    for (npos, nvel, nrad) in neighbours:
        d_sq = (npos[0] - pos[0]) ** 2 + (npos[1] - pos[1]) ** 2
        if d_sq < nd_sq:
            cand.append((d_sq, npos, nvel, nrad))
    cand.sort(key=lambda c: c[0])
    cand = cand[:max_neighbours]

    lines = []
    for (_, npos, nvel, nrad) in cand:
        rel_pos = (npos[0] - pos[0], npos[1] - pos[1])
        rel_vel = (vel[0] - nvel[0], vel[1] - nvel[1])
        dist_sq = _abs_sq(rel_pos)
        combined_radius = radius + nrad
        combined_radius_sq = combined_radius * combined_radius

        if dist_sq > combined_radius_sq:
            # No collision yet.
            w = (rel_vel[0] - inv_time_horizon * rel_pos[0],
                 rel_vel[1] - inv_time_horizon * rel_pos[1])
            w_len_sq = _abs_sq(w)
            dot1 = w[0] * rel_pos[0] + w[1] * rel_pos[1]

            if dot1 < 0.0 and dot1 * dot1 > combined_radius_sq * w_len_sq:
                # Project on cut-off circle.
                w_len = math.sqrt(w_len_sq)
                unit_w = (w[0] / w_len, w[1] / w_len)
                direction = (unit_w[1], -unit_w[0])
                u_mag = combined_radius * inv_time_horizon - w_len
                u = (u_mag * unit_w[0], u_mag * unit_w[1])
            else:
                # Project on legs.
                leg = math.sqrt(max(dist_sq - combined_radius_sq, 0.0))
                if _det(rel_pos, w) > 0.0:  # left leg
                    direction = ((rel_pos[0] * leg - rel_pos[1] * combined_radius) / dist_sq,
                                 (rel_pos[0] * combined_radius + rel_pos[1] * leg) / dist_sq)
                else:  # right leg
                    direction = (-(rel_pos[0] * leg + rel_pos[1] * combined_radius) / dist_sq,
                                 -(-rel_pos[0] * combined_radius + rel_pos[1] * leg) / dist_sq)
                dot2 = rel_vel[0] * direction[0] + rel_vel[1] * direction[1]
                u = (dot2 * direction[0] - rel_vel[0], dot2 * direction[1] - rel_vel[1])
        else:
            # Already colliding: use the time step instead of the horizon.
            w = (rel_vel[0] - inv_time_step * rel_pos[0],
                 rel_vel[1] - inv_time_step * rel_pos[1])
            w_len = math.sqrt(max(_abs_sq(w), EPSILON ** 2))
            unit_w = (w[0] / w_len, w[1] / w_len)
            direction = (unit_w[1], -unit_w[0])
            u_mag = combined_radius * inv_time_step - w_len
            u = (u_mag * unit_w[0], u_mag * unit_w[1])

        point = (vel[0] + responsibility * u[0], vel[1] + responsibility * u[1])
        lines.append((point, direction))

    line_fail, result = _linear_program2(lines, max_speed, pref_vel, False)
    if line_fail < len(lines):
        result = _linear_program3(lines, line_fail, max_speed, result)
    return result