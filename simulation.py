"""
Ransomware Detection - File Operation Simulator

Simulates realistic server-side file operations for an SME environment.
Closely follows the Aranyi et al. (2026) paper's methodology:
- 4 user profiles generating benign workload
- 5 ransomware families with characteristic attack patterns
- Operations aggregated per 1-second window

Feature vector (per 1-second window), matching the paper's 5 server-side
file operations:
    nc  : number of file creation operations
    nw  : number of file write operations
    nr  : number of file read operations
    nm  : number of file renaming operations
    nu  : number of file unlinking/deletion operations
    att : binary attack indicator (0=benign, 1=attack)
"""

import random
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class UserProfiles:
    """File operation ranges for each user profile (from paper)."""

    PROFILES = {
        "Secretary": {
            "create": (30, 60),
            "write": (40, 90),
            "read": (80, 160),
            "rename": (10, 20),
            "delete": (5, 20),
            "description": "Office document editing, filing",
        },
        "IT Admin": {
            "create": (30, 50),
            "write": (50, 100),
            "read": (100, 200),
            "rename": (10, 30),
            "delete": (10, 20),
            "description": "System management, backups",
        },
        "CEO": {
            "create": (0, 10),
            "write": (10, 40),
            "read": (30, 80),
            "rename": (0, 5),
            "delete": (0, 5),
            "description": "Light email and document work",
        },
        "Remote Worker": {
            "create": (20, 60),
            "write": (40, 100),
            "read": (80, 180),
            "rename": (10, 20),
            "delete": (5, 10),
            "description": "VPN-based file sync operations",
        },
    }


@dataclass
class RansomwareFamilies:
    """Attack patterns for 5 ransomware families."""

    FAMILIES = {
        "Ryuk": {
            "create": (80, 120),
            "write": (120, 200),
            "read": (180, 300),
            "rename": (40, 70),
            "delete": (30, 50),
            "duration_range": (10, 20),
            "ramp_up": 3,
            "description": "Targeted encryption, high-volume operations",
        },
        "WannaCry": {
            "create": (60, 100),
            "write": (100, 180),
            "read": (160, 280),
            "rename": (50, 80),
            "delete": (20, 40),
            "duration_range": (8, 15),
            "ramp_up": 2,
            "description": "Rapid spread, aggressive file operations",
        },
        "NotPetya": {
            "create": (70, 110),
            "write": (110, 190),
            "read": (170, 290),
            "rename": (30, 60),
            "delete": (40, 60),
            "duration_range": (12, 25),
            "ramp_up": 4,
            "description": "Destructive wiper disguised as ransomware",
        },
        "Lockbit": {
            "create": (90, 130),
            "write": (140, 220),
            "read": (200, 320),
            "rename": (50, 90),
            "delete": (35, 55),
            "duration_range": (5, 15),
            "ramp_up": 2,
            "description": "Fastest encryption, high throughput",
        },
        "Teslacrypt": {
            "create": (50, 90),
            "write": (90, 160),
            "read": (140, 250),
            "rename": (20, 50),
            "delete": (25, 45),
            "duration_range": (10, 20),
            "ramp_up": 3,
            "description": "Moderate speed, gaming file targets",
        },
    }


@dataclass
class SimulationState:
    """Tracks the current state of the simulation."""

    is_running: bool = False
    is_attack_active: bool = False
    current_user: str = "Secretary"
    current_attack_family: Optional[str] = None
    attack_seconds_remaining: int = 0
    attack_ramp_up_remaining: int = 0
    seconds_until_next_attack: int = 0
    total_seconds_simulated: int = 0
    total_attacks: int = 0
    attack_history: List[dict] = field(default_factory=list)
    simulation_speed: float = 1.0


class FileOperationSimulator:
    """
    Simulates server-side file operations for ransomware detection.

    Generates 1-second aggregated feature vectors describing five
    file-operation types: create, write, read, rename, delete.
    """

    FEATURE_KEYS = ("nc", "nw", "nr", "nm", "nu")

    def __init__(self, attack_interval_range=(15, 30)):
        self.user_profiles = UserProfiles.PROFILES
        self.ransomware_families = RansomwareFamilies.FAMILIES
        self.state = SimulationState()
        self.attack_interval_range = attack_interval_range
        self._reset_attack_timer()

    def _reset_attack_timer(self):
        """Set time until next attack."""
        self.state.seconds_until_next_attack = random.randint(
            self.attack_interval_range[0], self.attack_interval_range[1]
        )

    def _pick_random_user(self) -> str:
        """Randomly select a user profile."""
        return random.choice(list(self.user_profiles.keys()))

    def _pick_random_family(self) -> str:
        """Randomly select a ransomware family."""
        return random.choice(list(self.ransomware_families.keys()))

    def _ops_from_profile(self, profile: dict) -> Tuple[int, int, int, int, int]:
        """Generate a 5-tuple of file operations from a profile dict."""
        return (
            random.randint(*profile["create"]),
            random.randint(*profile["write"]),
            random.randint(*profile["read"]),
            random.randint(*profile["rename"]),
            random.randint(*profile["delete"]),
        )

    def _generate_benign_ops(self, user: str) -> Tuple[int, int, int, int, int]:
        """Generate benign file operation counts for a user profile."""
        profile = self.user_profiles[user]
        return self._ops_from_profile(profile)

    def _generate_attack_ops(self, family: str, progress: float) -> Tuple[int, int, int, int, int]:
        """
        Generate attack file operation counts with ramp-up effect.

        Args:
            family: Ransomware family name
            progress: Float 0.0-1.0 indicating attack progress
        """
        fam = self.ransomware_families[family]

        # Ramp-up: operations increase during first few seconds
        ramp_factor = min(1.0, progress / 0.3) if fam["ramp_up"] > 0 else 1.0

        base_nc, base_nw, base_nr, base_nm, base_nu = self._ops_from_profile(fam)

        nc = int(base_nc * ramp_factor)
        nw = int(base_nw * ramp_factor)
        nr = int(base_nr * ramp_factor)
        nm = int(base_nm * ramp_factor)
        nu = int(base_nu * ramp_factor)

        # Enforce minimum activity for attack windows
        min_activity = max(3, int(ramp_factor * 10))
        nc = max(nc, min_activity)
        nw = max(nw, min_activity)
        nr = max(nr, min_activity)
        nm = max(nm, min_activity // 2)
        nu = max(nu, min_activity // 2)

        return nc, nw, nr, nm, nu

    def generate_next_tick(self) -> dict:
        """
        Generate the next 1-second tick of file operations.

        Returns a dict with keys: nc, nw, nr, nm, nu, att, user, family,
        description, timestamp, is_attack.
        """
        self.state.total_seconds_simulated += 1

        result = {
            "timestamp": self.state.total_seconds_simulated,
            "nc": 0, "nw": 0, "nr": 0, "nm": 0, "nu": 0,
            "att": 0, "user": "", "family": None,
            "description": "", "is_attack": False,
        }

        if self.state.is_attack_active:
            self.state.attack_seconds_remaining -= 1
            total_duration = self.ransomware_families[self.state.current_attack_family]["duration_range"]
            elapsed = max(1, (total_duration[0] + total_duration[1]) // 2 - self.state.attack_seconds_remaining)
            progress = elapsed / max(1, (total_duration[0] + total_duration[1]) // 2)

            nc, nw, nr, nm, nu = self._generate_attack_ops(self.state.current_attack_family, progress)

            if random.random() < 0.3:
                b_nc, b_nw, b_nr, b_nm, b_nu = self._generate_benign_ops(self.state.current_user)
                nc += b_nc // 3
                nw += b_nw // 3
                nr += b_nr // 3
                nm += b_nm // 3
                nu += b_nu // 3

            result.update({
                "nc": nc, "nw": nw, "nr": nr, "nm": nm, "nu": nu,
                "att": 1, "user": self.state.current_user,
                "family": self.state.current_attack_family,
                "description": self.ransomware_families[self.state.current_attack_family]["description"],
                "is_attack": True,
            })

            if self.state.attack_seconds_remaining <= 0:
                self.state.is_attack_active = False
                self.state.current_attack_family = None
                self._reset_attack_timer()

        else:
            nc, nw, nr, nm, nu = self._generate_benign_ops(self.state.current_user)
            result.update({
                "nc": nc, "nw": nw, "nr": nr, "nm": nm, "nu": nu,
                "att": 0, "user": self.state.current_user,
                "description": self.user_profiles[self.state.current_user]["description"],
            })

            self.state.seconds_until_next_attack -= 1
            if self.state.seconds_until_next_attack <= 0:
                family = self._pick_random_family()
                duration_range = self.ransomware_families[family]["duration_range"]
                duration = random.randint(*duration_range)

                self.state.is_attack_active = True
                self.state.current_attack_family = family
                self.state.attack_seconds_remaining = duration
                self.state.total_attacks += 1

                self.state.attack_history.append({
                    "attack_id": self.state.total_attacks,
                    "family": family,
                    "start_second": self.state.total_seconds_simulated,
                    "duration": duration,
                })

                nc, nw, nr, nm, nu = self._generate_attack_ops(family, 0.0)
                result.update({
                    "nc": nc, "nw": nw, "nr": nr, "nm": nm, "nu": nu,
                    "att": 1, "family": family,
                    "description": self.ransomware_families[family]["description"],
                    "is_attack": True,
                })
                self.state.attack_seconds_remaining -= 1

                if self.state.attack_seconds_remaining <= 0:
                    self.state.is_attack_active = False
                    self.state.current_attack_family = None
                    self._reset_attack_timer()

            if random.random() < 0.1:
                self.state.current_user = self._pick_random_user()

        return result

    def start(self):
        """Start the simulation."""
        self.state.is_running = True
        self.state.current_user = self._pick_random_user()
        self._reset_attack_timer()

    def stop(self):
        """Stop the simulation."""
        self.state.is_running = False

    def get_state(self) -> dict:
        """Get current simulation state."""
        return {
            "is_running": self.state.is_running,
            "is_attack_active": self.state.is_attack_active,
            "current_user": self.state.current_user,
            "current_attack_family": self.state.current_attack_family,
            "attack_seconds_remaining": self.state.attack_seconds_remaining,
            "seconds_until_next_attack": self.state.seconds_until_next_attack,
            "total_seconds_simulated": self.state.total_seconds_simulated,
            "total_attacks": self.state.total_attacks,
            "attack_history": self.state.attack_history[-10:],
            "simulation_speed": self.state.simulation_speed,
        }

    def set_speed(self, speed: float):
        """Set simulation speed multiplier."""
        self.state.simulation_speed = max(0.1, min(5.0, speed))

    def force_attack(self, family: str = None):
        """Force an immediate attack for demo purposes."""
        if self.state.is_attack_active:
            return False

        if family is None:
            family = self._pick_random_family()
        elif family not in self.ransomware_families:
            return False

        duration_range = self.ransomware_families[family]["duration_range"]
        duration = random.randint(*duration_range)

        self.state.is_attack_active = True
        self.state.current_attack_family = family
        self.state.attack_seconds_remaining = duration
        self.state.total_attacks += 1

        self.state.attack_history.append({
            "attack_id": self.state.total_attacks,
            "family": family,
            "start_second": self.state.total_seconds_simulated,
            "duration": duration,
        })

        return True


def _profile_ops(profile: dict) -> Tuple[int, int, int, int, int]:
    """Random file operation tuple for a profile dict (training helper)."""
    return (
        random.randint(*profile["create"]),
        random.randint(*profile["write"]),
        random.randint(*profile["read"]),
        random.randint(*profile["rename"]),
        random.randint(*profile["delete"]),
    )


def _attack_ops(fam: dict, progress: float) -> Tuple[int, int, int, int, int]:
    """Random attack operation tuple with ramp-up (training helper)."""
    ramp_factor = min(1.0, progress / 0.3) if fam["ramp_up"] > 0 else 1.0
    nc, nw, nr, nm, nu = _profile_ops(fam)
    return (
        int(nc * ramp_factor),
        int(nw * ramp_factor),
        int(nr * ramp_factor),
        int(nm * ramp_factor),
        int(nu * ramp_factor),
    )


def generate_training_data(num_samples: int = 10000) -> List[dict]:
    """
    Generate synthetic training data for ML model development.

    Creates a balanced dataset of benign and attack patterns based on
    the paper's simulation parameters.
    """
    data = []

    benign_count = 0
    attack_count = 0
    target_per_class = num_samples // 2

    # Phase 1: Benign-only data
    users = list(UserProfiles.PROFILES.keys())

    # Add explicit idle/zero samples as benign
    idle_samples = target_per_class // 10
    for _ in range(idle_samples):
        data.append({"nc": 0, "nw": 0, "nr": 0, "nm": 0, "nu": 0, "att": 0})
        benign_count += 1

    # Add low-activity benign samples
    while benign_count < target_per_class:
        user = random.choice(users)
        profile = UserProfiles.PROFILES[user]
        nc, nw, nr, nm, nu = _profile_ops(profile)

        # Add occasional heavy-operation spikes
        if random.random() < 0.05:
            nc = int(nc * random.uniform(1.5, 2.5))
            nw = int(nw * random.uniform(1.3, 2.0))
            nr = int(nr * random.uniform(1.2, 1.8))
            nm = int(nm * random.uniform(1.3, 2.0))
            nu = int(nu * random.uniform(1.2, 1.8))

        data.append({
            "nc": nc, "nw": nw, "nr": nr, "nm": nm, "nu": nu, "att": 0,
        })
        benign_count += 1

    # Phase 2: Attack data with ramp-up
    families = list(RansomwareFamilies.FAMILIES.keys())
    while attack_count < target_per_class:
        family = random.choice(families)
        fam = RansomwareFamilies.FAMILIES[family]
        duration = random.randint(*fam["duration_range"])

        for second in range(duration):
            if attack_count >= target_per_class:
                break

            progress = second / max(1, duration)
            ramp_factor = min(1.0, progress / 0.3) if fam["ramp_up"] > 0 else 1.0
            min_activity = max(3, int(ramp_factor * 10))

            nc, nw, nr, nm, nu = _attack_ops(fam, progress)

            nc = max(nc, min_activity)
            nw = max(nw, min_activity)
            nr = max(nr, min_activity)
            nm = max(nm, min_activity // 2)
            nu = max(nu, min_activity // 2)

            # Add noise
            if random.random() < 0.2:
                user = random.choice(users)
                profile = UserProfiles.PROFILES[user]
                bn_c, bn_w, bn_r, bn_m, bn_u = _profile_ops(profile)
                nc += bn_c // 4
                nw += bn_w // 4
                nr += bn_r // 4
                nm += bn_m // 4
                nu += bn_u // 4

            data.append({
                "nc": max(1, nc), "nw": max(1, nw), "nr": max(1, nr),
                "nm": max(0, nm), "nu": max(0, nu), "att": 1,
            })
            attack_count += 1

    # Phase 3: Transition data (attack ramp-up and wind-down)
    transitions = max(1, num_samples // 20)
    for _ in range(transitions):
        family = random.choice(families)
        fam = RansomwareFamilies.FAMILIES[family]

        # Ramp-up transition
        for i in range(5):
            factor = (i + 1) / 5.0
            uc, uw, ur, um, uu = _profile_ops(fam)
            data.append({
                "nc": max(3, int(uc * factor * 0.7)),
                "nw": max(2, int(uw * factor * 0.7)),
                "nr": max(2, int(ur * factor * 0.7)),
                "nm": max(1, int(um * factor * 0.7)),
                "nu": max(1, int(uu * factor * 0.7)),
                "att": 1,
            })

    random.shuffle(data)
    return data


if __name__ == "__main__":
    # Quick test
    sim = FileOperationSimulator(attack_interval_range=(5, 10))
    sim.start()

    print("Generating 30 seconds of simulated data...")
    for i in range(30):
        tick = sim.generate_next_tick()
        marker = " *** ATTACK ***" if tick["is_attack"] else ""
        print(
            f"  [{tick['timestamp']:3d}s] nc={tick['nc']:3d} nw={tick['nw']:3d} "
            f"nr={tick['nr']:3d} nm={tick['nm']:3d} nu={tick['nu']:3d} "
            f"att={tick['att']} user={tick['user']:15s}{marker}"
        )

    print(f"\nTotal attacks: {sim.state.total_attacks}")
    print(f"Attack history: {sim.state.attack_history}")
