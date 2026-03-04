import torch

import nrm.dataset.se3 as se3
from nrm.dataset.self_collision import collision_check, EPS
from nrm.dataset.kinematics import pure_analytical_inverse_kinematics, analytical_inverse_kinematics, \
    forward_kinematics, numerical_inverse_kinematics
from nrm.dataset.morphology import sample_morph, get_joint_limits

torch.set_printoptions(sci_mode=False, precision=2)
torch.set_default_dtype(torch.float64)


def test_pure_analytical_inverse_kinematics():
    torch.manual_seed(1)
    n_samples = 1000
    n_robots = 10
    morphs = sample_morph(n_robots, 6, True)
    for morph_idx, morph in enumerate(morphs):
        joint_limits = get_joint_limits(morph).unsqueeze(0).expand(n_samples, -1, -1)
        morph = morph.unsqueeze(0).expand(n_samples, -1, -1)
        joints = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + joint_limits[
            ..., 1:2]
        poses = forward_kinematics(morph, joints)[:, -1, :, :]
        solutions = pure_analytical_inverse_kinematics(morph[0], poses)

        for i, solution in enumerate(solutions):
            if solution.shape[0] == 0:
                assert False, (f"IK failed to find a solution for pose \n{poses[i]}\n, "
                               f"joints \n{joints[i, :, 0]}\n and morph \n{morph[0]}\n")

            dist = torch.arctan2(torch.sin(solution[..., 0] - joints[i, :, 0]),
                                 torch.cos(solution[..., 0] - joints[i, :, 0]))
            dist = torch.norm(dist, dim=-1)
            assert torch.min(dist).item() < torch.pi / 100, (
                f"IK failed to reconstruct the joints \n {joints[i, :, 0]}\n"
                f"It only found \n{solution[..., 0]}\n"
                f"Instead of reaching pose \n{poses[i]}\n"
                f"It reached only \n{forward_kinematics(morph[0].unsqueeze(0).expand(solution.shape[0], -1, -1), solution)[:, -1, :, :]}\n"
                f"For morph \n{morph[0]}\n"
                )


def test_analytical_inverse_kinematics():
    torch.manual_seed(1)
    n_samples = 1000
    n_robots = 10
    morphs = sample_morph(n_robots, 6, True)

    for i, morph in enumerate(morphs):
        joint_limits = get_joint_limits(morph).unsqueeze(0).expand(n_samples, -1, -1)
        morph = morph.unsqueeze(0).expand(n_samples, -1, -1)
        joints = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + joint_limits[
            ..., 1:2]
        poses = forward_kinematics(morph, joints)
        self_collision = collision_check(morph, poses)
        poses = poses[:, -1, :, :]
        joints, manipulability = analytical_inverse_kinematics(morph[0], poses)

        mask = manipulability != -1
        assert torch.all(self_collision[~mask]), (f"IK does not find solutions for joints "
                                                  f"\n{joints[~self_collision[~mask]]}\n "
                                                  f"and poses"
                                                  f"\n{poses[~self_collision[~mask]]}\n"
                                                  f"given morph"
                                                  f"\n{morph[0]}\n")

        morph = morph[0].unsqueeze(0).expand(mask.sum(), -1, -1)
        ik_poses = forward_kinematics(morph, joints[mask])
        assert se3.distance(poses[mask], ik_poses[:, -1, :, :]).max() < EPS, f"IK finds wrong solutions"
        ik_self_collisions = collision_check(morph, ik_poses)
        assert torch.all(~ik_self_collisions), f"IK solution has self-collisions"


def test_numerical_inverse_kinematics():
    torch.manual_seed(1)
    n_samples = 100
    n_robots = 5

    morphs_analytical = sample_morph(n_robots, 6, True)
    morphs_general = sample_morph(n_robots, 6, False)

    for j, morphs in enumerate([morphs_analytical, morphs_general]):
        for i, morph in enumerate(morphs):
            joint_limits = get_joint_limits(morph).unsqueeze(0).expand(n_samples, -1, -1)
            morph = morph.unsqueeze(0).expand(n_samples, -1, -1)

            joints_gt = torch.rand(*joint_limits.shape[:-1], 1, device=morph.device) * joint_limits[..., 0:1] + \
                        joint_limits[..., 1:2]
            poses = forward_kinematics(morph, joints_gt)
            self_collision = collision_check(morph, poses)
            poses = poses[~self_collision, -1, :, :]
            morph = morph[~self_collision]

            if poses.shape[0] == 0:
                continue

            joints_num, man_num = numerical_inverse_kinematics(morph[0], poses)

            mask_num_success = man_num != -1

            if mask_num_success.any():
                ik_poses_num = forward_kinematics(morph[mask_num_success], joints_num[mask_num_success])
                pose_error = se3.distance(poses[mask_num_success], ik_poses_num[:, -1, :, :])
                assert pose_error.max() < EPS, f"Numerical IK claims success but pose error {pose_error.max()} > {EPS}"

                ik_collisions_num = collision_check(morph[mask_num_success], ik_poses_num)
                assert torch.all(~ik_collisions_num), "Numerical IK claims success but returned a self-colliding configuration."

            if j== 0:
                joints_ana, man_ana = analytical_inverse_kinematics(morph[0], poses)
                mask_ana_success = man_ana != -1
                both_success = mask_ana_success & mask_num_success

                if mask_ana_success.sum() > 0:
                    recall = both_success.sum() / mask_ana_success.sum()
                    print(f"Robot {i}: Numerical IK Recall vs Analytical: {recall:.2%} "
                          f"({both_success.sum()}/{mask_ana_success.sum()})")

                    # Warn if numerical solver is performing terrible (e.g. < 50% of what analytical finds)
                    # Note: Numerical IK is local and random seeded, so it might not find 100%,
                    # but it shouldn't fail completely on solvable poses.
                    if recall < 0.1:
                        print(f"Warning: Numerical IK success rate is suspiciously low for Robot {i}")
