import numpy as np


class WeightCalculator:
    """Compute policy aggregation weights from approach and surround phases."""

    def __init__(self, sigma_d=20, capture_radius=40):
        self.sigma_d = sigma_d
        self.capture_radius = capture_radius

    def compute_approach_weights(self, approach_start_info, approach_end_info, time_steps):
        """Return normalized weights from each pursuer's approach-stage progress."""
        weights = []
        for start, end in zip(approach_start_info, approach_end_info):
            delta_d = start['distance'] - end['distance']
            delta_theta = start['angle'] - end['angle']
            delta_theta = self.wrap_to_pi(delta_theta)

            distance_term = np.exp(max(0.0, delta_d) / (self.sigma_d + 1e-6))
            angle_term = max(0, np.cos(delta_theta))
            time_term = time_steps

            w = (distance_term * angle_term) / (time_term + 1e-6)
            weights.append(w)

        total = sum(weights)
        if total > 1e-8:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
        return weights

    def compute_surround_weights(self, surround_start_info, surround_end_info, num_agents):
        """Return normalized weights from each pursuer's surround-stage progress."""
        weights = []
        for idx, (start, end) in enumerate(zip(surround_start_info, surround_end_info)):
            expected_angle = (2 * np.pi / num_agents) * idx

            start_cep = 1.0 / (
                abs(start['distance'] - self.capture_radius)
                + abs(self.wrap_to_pi(start['angle'] - expected_angle))
                + 1e-6
            )
            end_cep = 1.0 / (
                abs(end['distance'] - self.capture_radius)
                + abs(self.wrap_to_pi(end['angle'] - expected_angle))
                + 1e-6
            )
            delta_cep = end_cep - start_cep

            weights.append(max(0, delta_cep))

        total = sum(weights) + 1e-6
        weights = [w / total for w in weights]
        return weights

    def combine_approach_and_surround_weights(approach_weights, surround_weights):
        """Normalize, add, and re-normalize approach and surround weights."""
        total_approach = sum(approach_weights)
        if total_approach > 0:
            approach_weights = [w / total_approach for w in approach_weights]
        else:
            approach_weights = [1.0 / len(approach_weights)] * len(approach_weights)

        total_surround = sum(surround_weights)
        if total_surround > 0:
            surround_weights = [w / total_surround for w in surround_weights]
        else:
            surround_weights = [1.0 / len(surround_weights)] * len(surround_weights)

        combined_weights = [a + s for a, s in zip(approach_weights, surround_weights)]

        total_combined = sum(combined_weights)
        if total_combined > 0:
            normalized_weights = [w / total_combined for w in combined_weights]
        else:
            normalized_weights = [1.0 / len(combined_weights)] * len(combined_weights)

        return normalized_weights

    @staticmethod
    def wrap_to_pi(angle):
        """Wrap an angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
