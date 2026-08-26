"""
Tests for the file operation simulator module.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from simulation import FileOperationSimulator, generate_training_data, RansomwareFamilies, UserProfiles


class TestUserProfiles:
    """Test user profile definitions."""

    def test_profiles_exist(self):
        assert len(UserProfiles.PROFILES) > 0

    def test_profile_structure(self):
        for name, profile in UserProfiles.PROFILES.items():
            assert "create" in profile
            assert "rename" in profile
            assert "delete" in profile
            assert profile["create"][0] <= profile["create"][1]
            assert profile["rename"][0] <= profile["rename"][1]
            assert profile["delete"][0] <= profile["delete"][1]


class TestRansomwareFamilies:
    """Test ransomware family definitions."""

    def test_families_exist(self):
        assert len(RansomwareFamilies.FAMILIES) > 0

    def test_family_structure(self):
        for name, fam in RansomwareFamilies.FAMILIES.items():
            assert "create" in fam
            assert "rename" in fam
            assert "delete" in fam
            assert "duration_range" in fam
            assert "ramp_up" in fam


class TestFileOperationSimulator:
    """Test the file operation simulator."""

    def test_start_stop(self):
        sim = FileOperationSimulator()
        sim.start()
        assert sim.state is not None
        sim.stop()

    def test_generate_tick(self):
        sim = FileOperationSimulator()
        sim.start()
        tick = sim.generate_next_tick()
        assert "nc" in tick
        assert "nr" in tick
        assert "nu" in tick
        assert "att" in tick
        assert tick["nc"] >= 0
        assert tick["nr"] >= 0
        assert tick["nu"] >= 0
        sim.stop()

    def test_force_attack(self):
        sim = FileOperationSimulator()
        sim.start()
        result = sim.force_attack()
        assert isinstance(result, bool)
        sim.stop()

    def test_speed_control(self):
        sim = FileOperationSimulator()
        sim.set_speed(3.0)
        assert sim.state.simulation_speed == 3.0


class TestTrainingData:
    """Test training data generation."""

    def test_generate_data(self):
        data = generate_training_data(num_samples=100)
        assert len(data) > 0
        assert len(data) >= 100

    def test_data_structure(self):
        data = generate_training_data(num_samples=50)
        for sample in data:
            assert "nc" in sample
            assert "nr" in sample
            assert "nu" in sample
            assert "att" in sample
            assert sample["att"] in [0, 1]

    def test_no_zero_attack_samples(self):
        data = generate_training_data(num_samples=500)
        for sample in data:
            if sample["nc"] == 0 and sample["nr"] == 0 and sample["nu"] == 0:
                assert sample["att"] == 0, "Zero-value samples must be benign"
