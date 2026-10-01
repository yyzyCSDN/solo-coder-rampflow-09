import unittest
from datetime import datetime, timezone

from rampflow.assistance_handoff import (
    AssistanceEquipment,
    AssistanceHandoffManager,
    AssistanceJourney,
    AssistanceLeg,
    HandoffScan,
    HandoffState,
)


UTC = timezone.utc
AT = lambda minute: datetime(2026, 10, 1, 10, minute, tzinfo=UTC)


def journey() -> AssistanceJourney:
    return AssistanceJourney(
        "J1",
        "P1",
        "CA123",
        (
            AssistanceLeg(
                "gate-to-bus",
                "gate-agent",
                "shuttle-driver",
                "G12",
                frozenset({"wheelchair", "shuttle"}),
                AT(5),
                AT(15),
            ),
            AssistanceLeg(
                "bus-to-platform",
                "shuttle-driver",
                "ambulift-operator",
                "remote-stand-9",
                frozenset({"shuttle", "ambulift"}),
                AT(20),
                AT(35),
            ),
        ),
    )


def manager() -> AssistanceHandoffManager:
    return AssistanceHandoffManager(
        [
            AssistanceEquipment("W1", "wheelchair"),
            AssistanceEquipment("B1", "shuttle"),
            AssistanceEquipment("L1", "ambulift"),
        ]
    )


class AssistanceHandoffTests(unittest.TestCase):
    def test_connected_chain_shows_responsibility_and_remaining_time(self):
        mgr = manager()
        mgr.register(journey())

        waiting = mgr.status("J1", AT(0))
        self.assertEqual(waiting.state, HandoffState.WAITING)
        self.assertEqual(waiting.current_leg_id, "gate-to-bus")
        self.assertEqual(waiting.current_responsible_party, "gate-agent")
        self.assertEqual(waiting.awaiting_party, "shuttle-driver")
        self.assertEqual(waiting.remaining_minutes, 15)
        self.assertEqual(waiting.legs[1].state, HandoffState.BLOCKED)
        self.assertFalse(waiting.connected)

        first = mgr.record_scan(
            HandoffScan(
                "S1",
                "J1",
                "gate-to-bus",
                AT(6),
                "shuttle-driver",
                ("W1", "B1"),
            )
        )
        self.assertTrue(first.accepted)
        self.assertEqual(first.code, "ok")
        self.assertTrue(first.details["counted"])
        self.assertEqual(first.details["current_responsible_party"], "shuttle-driver")
        self.assertEqual(first.details["awaiting_party"], "ambulift-operator")
        self.assertEqual(first.details["remaining_minutes"], 29)

        after_first = mgr.status("J1", AT(6))
        self.assertEqual(after_first.state, HandoffState.WAITING)
        self.assertEqual(after_first.current_responsible_party, "shuttle-driver")
        self.assertEqual(after_first.current_leg_id, "bus-to-platform")
        self.assertEqual(after_first.completed_legs, 1)
        self.assertEqual(after_first.legs[0].state, HandoffState.COMPLETE)

        second = mgr.record_scan(
            HandoffScan(
                "S2",
                "J1",
                "bus-to-platform",
                AT(7),
                "ambulift-operator",
                ("B1", "L1"),
            )
        )
        self.assertTrue(second.accepted)
        self.assertEqual(mgr.status("J1", AT(7)).state, HandoffState.COMPLETE)
        self.assertTrue(mgr.status("J1", AT(7)).connected)
        self.assertEqual(mgr.status("J1", AT(7)).current_responsible_party, "ambulift-operator")

    def test_repeated_scan_is_idempotent_and_not_counted_twice(self):
        mgr = manager()
        mgr.register(journey())
        scan = HandoffScan(
            "S1",
            "J1",
            "gate-to-bus",
            AT(4),
            "shuttle-driver",
            ("W1", "B1"),
        )

        first = mgr.record_scan(scan)
        replay = mgr.record_scan(scan)
        another = mgr.record_scan(
            HandoffScan(
                "S1-repeat-device",
                "J1",
                "gate-to-bus",
                AT(4),
                "shuttle-driver",
                ("W1", "B1"),
            )
        )

        self.assertIs(first, replay)
        self.assertTrue(replay.details["counted"])
        self.assertTrue(another.accepted)
        self.assertEqual(another.code, "duplicate")
        self.assertFalse(another.details["counted"])
        self.assertEqual(mgr.status("J1", AT(4)).completed_legs, 1)

    def test_later_leg_cannot_be_completed_before_current_gap(self):
        mgr = manager()
        mgr.register(journey())

        decision = mgr.record_scan(
            HandoffScan(
                "S2",
                "J1",
                "bus-to-platform",
                AT(10),
                "ambulift-operator",
                ("B1", "L1"),
            )
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "chain-gap")
        self.assertEqual(decision.details["current_leg_id"], "gate-to-bus")
        self.assertEqual(decision.details["responsible_party"], "gate-agent")
        self.assertEqual(decision.details["remaining_minutes"], 5)
        self.assertEqual(mgr.status("J1", AT(10)).state, HandoffState.OVERDUE)
        self.assertEqual(mgr.status("J1", AT(10)).completed_legs, 0)

    def test_missing_or_unserviceable_equipment_blocks_acceptance(self):
        mgr = manager()
        mgr.register(journey())

        missing = mgr.record_scan(
            HandoffScan(
                "S-missing",
                "J1",
                "gate-to-bus",
                AT(3),
                "shuttle-driver",
                ("W1",),
            )
        )
        self.assertFalse(missing.accepted)
        self.assertEqual(missing.code, "equipment-missing")
        self.assertEqual(missing.details["missing_equipment"], ("shuttle",))
        self.assertEqual(mgr.status("J1", AT(3)).completed_legs, 0)

        mgr.equipment["B1"] = AssistanceEquipment("B1", "shuttle", available=False)
        broken = mgr.record_scan(
            HandoffScan(
                "S-broken",
                "J1",
                "gate-to-bus",
                AT(3),
                "shuttle-driver",
                ("W1", "B1"),
            )
        )
        self.assertFalse(broken.accepted)
        self.assertEqual(broken.code, "equipment-unavailable")
        self.assertEqual(broken.details["unavailable_equipment"], ("B1",))

    def test_wrong_party_and_out_of_order_scans_are_rejected(self):
        mgr = manager()
        mgr.register(journey())

        wrong_party = mgr.record_scan(
            HandoffScan(
                "S-wrong",
                "J1",
                "gate-to-bus",
                AT(4),
                "ambulift-operator",
                ("W1", "B1"),
            )
        )
        self.assertEqual(wrong_party.code, "wrong-party")
        self.assertEqual(wrong_party.details["expected_party"], "shuttle-driver")

        first = mgr.record_scan(
            HandoffScan(
                "S1",
                "J1",
                "gate-to-bus",
                AT(6),
                "shuttle-driver",
                ("W1", "B1"),
            )
        )
        self.assertTrue(first.accepted)
        second = mgr.record_scan(
            HandoffScan(
                "S2",
                "J1",
                "bus-to-platform",
                AT(5),
                "ambulift-operator",
                ("B1", "L1"),
            )
        )
        self.assertFalse(second.accepted)
        self.assertEqual(second.code, "scan-order")

    def test_overdue_and_expired_status_expose_current_owner_and_remedy_deadline(self):
        mgr = manager()
        mgr.register(journey())

        overdue = mgr.status("J1", AT(10))
        self.assertEqual(overdue.state, HandoffState.OVERDUE)
        self.assertEqual(overdue.current_responsible_party, "gate-agent")
        self.assertEqual(overdue.remaining_minutes, 5)

        expired_scan = mgr.record_scan(
            HandoffScan(
                "S-late",
                "J1",
                "gate-to-bus",
                AT(16),
                "shuttle-driver",
                ("W1", "B1"),
            )
        )
        self.assertFalse(expired_scan.accepted)
        self.assertEqual(expired_scan.code, "remedy-expired")
        self.assertEqual(expired_scan.details["state"], HandoffState.EXPIRED)
        self.assertEqual(expired_scan.details["remaining_minutes"], -1)
        self.assertEqual(mgr.status("J1", AT(16)).state, HandoffState.EXPIRED)

    def test_chain_construction_must_connect_responsibility_parties(self):
        with self.assertRaises(ValueError):
            AssistanceJourney(
                "J2",
                "P2",
                "CA456",
                (
                    AssistanceLeg(
                        "gate-to-bus",
                        "gate-agent",
                        "shuttle-driver",
                        "G12",
                        frozenset({"wheelchair"}),
                        AT(5),
                        AT(15),
                    ),
                    AssistanceLeg(
                        "bus-to-platform",
                        "different-driver",
                        "ambulift-operator",
                        "stand",
                        frozenset({"ambulift"}),
                        AT(20),
                        AT(35),
                    ),
                ),
            )

    def test_scan_id_replay_cannot_change_identity(self):
        mgr = manager()
        mgr.register(journey())
        scan = HandoffScan(
            "S1",
            "J1",
            "gate-to-bus",
            AT(4),
            "shuttle-driver",
            ("W1", "B1"),
        )
        mgr.record_scan(scan)

        with self.assertRaises(ValueError):
            mgr.record_scan(
                HandoffScan(
                    "S1",
                    "J1",
                    "gate-to-bus",
                    AT(4),
                    "shuttle-driver",
                    ("W1",),
                )
            )


if __name__ == "__main__":
    unittest.main()
