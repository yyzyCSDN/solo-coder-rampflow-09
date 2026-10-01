import unittest
from datetime import datetime, timezone

from rampflow.prm_handover import (
    ChainState,
    HandoverSegment,
    PrmHandoverManager,
    PrmJourney,
    SegmentState,
)
from rampflow.timeutil import Window
from rampflow.workflows.control import RampWorkflowControl

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)


def chain():
    return (
        HandoverSegment(
            "gate-to-bus", "gate-agent", "bus-driver",
            frozenset({"wheelchair", "accessible-bus"}),
            Window(D(10, 0), D(10, 10)), D(10, 20),
        ),
        HandoverSegment(
            "bus-to-lift", "bus-driver", "lift-operator",
            frozenset({"ambulift"}),
            Window(D(10, 15), D(10, 25)), D(10, 35),
        ),
        HandoverSegment(
            "lift-to-aircraft", "lift-operator", "cabin-crew",
            frozenset(),
            Window(D(10, 30), D(10, 40)), D(10, 45),
        ),
    )


def journey():
    return PrmJourney("PAX1", "FL100", chain())


class PrmHandoverTests(unittest.TestCase):
    def test_full_chain_delivers_passenger(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        manager.scan("PAX1", "FL100", "gate-to-bus", "s1", D(10, 5), equipment=("wheelchair", "accessible-bus"))
        manager.scan("PAX1", "FL100", "bus-to-lift", "s2", D(10, 16), equipment=("ambulift",))
        manager.scan("PAX1", "FL100", "lift-to-aircraft", "s3", D(10, 32))
        report = manager.status("PAX1", "FL100", D(10, 33))
        self.assertEqual(report.state, ChainState.COMPLETED)
        self.assertIsNone(report.current_segment)
        self.assertEqual(report.responsible_party, "cabin-crew")
        self.assertIsNone(report.remaining_minutes)
        self.assertTrue(all(v.state == SegmentState.COMPLETED for v in report.segments))
        self.assertEqual(manager.breach_report(D(10, 33)), ())

    def test_repeated_scan_is_not_counted_twice(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        first = manager.scan("PAX1", "FL100", "gate-to-bus", "s1", D(10, 5), equipment=("wheelchair", "accessible-bus"))
        again = manager.scan("PAX1", "FL100", "gate-to-bus", "s1", D(10, 6), equipment=("wheelchair", "accessible-bus"))
        self.assertIs(first, again)
        self.assertEqual(again.recorded_at, D(10, 5))
        report = manager.status("PAX1", "FL100", D(10, 7))
        self.assertEqual(report.version, 2)
        self.assertEqual(report.current_segment, "bus-to-lift")
        completed = [v for v in report.segments if v.state == SegmentState.COMPLETED]
        self.assertEqual(len(completed), 1)

    def test_new_scan_id_on_completed_segment_rejected(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        manager.scan("PAX1", "FL100", "gate-to-bus", "s1", D(10, 5), equipment=("wheelchair", "accessible-bus"))
        dupe = manager.scan("PAX1", "FL100", "gate-to-bus", "s2", D(10, 6), equipment=("wheelchair", "accessible-bus"))
        self.assertFalse(dupe.accepted)
        self.assertEqual(dupe.code, "duplicate-segment")
        report = manager.status("PAX1", "FL100", D(10, 7))
        self.assertEqual(report.version, 2)
        nxt = manager.scan("PAX1", "FL100", "bus-to-lift", "s3", D(10, 16), equipment=("ambulift",))
        self.assertTrue(nxt.accepted)

    def test_segments_must_connect_in_order(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        skipped = manager.scan("PAX1", "FL100", "bus-to-lift", "s1", D(10, 16), equipment=("ambulift",))
        self.assertFalse(skipped.accepted)
        self.assertEqual(skipped.code, "out-of-order")
        self.assertEqual(skipped.details["awaiting"], "gate-to-bus")

    def test_missing_equipment_blocks_handover(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        denied = manager.scan("PAX1", "FL100", "gate-to-bus", "s1", D(10, 5), equipment=("wheelchair",))
        self.assertFalse(denied.accepted)
        self.assertEqual(denied.code, "equipment-missing")
        self.assertEqual(denied.details["missing"], ("accessible-bus",))
        ok = manager.scan("PAX1", "FL100", "gate-to-bus", "s2", D(10, 6), equipment=("wheelchair", "accessible-bus"))
        self.assertTrue(ok.accepted)

    def test_unconnected_segment_exposes_owner_and_deadline(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        report = manager.status("PAX1", "FL100", D(10, 12))
        self.assertEqual(report.state, ChainState.OVERDUE)
        self.assertEqual(report.current_segment, "gate-to-bus")
        self.assertEqual(report.responsible_party, "gate-agent")
        self.assertEqual(report.expected_receiver, "bus-driver")
        self.assertEqual(report.required_equipment, ("accessible-bus", "wheelchair"))
        self.assertEqual(report.recover_by, D(10, 20))
        self.assertEqual(report.remaining_minutes, 8)
        broken = manager.status("PAX1", "FL100", D(10, 25))
        self.assertEqual(broken.state, ChainState.BROKEN)
        self.assertEqual(broken.responsible_party, "gate-agent")
        self.assertEqual(broken.remaining_minutes, -5)

    def test_scan_after_recovery_deadline_rejected(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        late = manager.scan("PAX1", "FL100", "gate-to-bus", "s9", D(10, 21), equipment=("wheelchair", "accessible-bus"))
        self.assertFalse(late.accepted)
        self.assertEqual(late.code, "recovery-expired")
        self.assertEqual(manager.status("PAX1", "FL100", D(10, 21)).state, ChainState.BROKEN)

    def test_scan_id_cannot_move_to_another_segment(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        manager.scan("PAX1", "FL100", "gate-to-bus", "s1", D(10, 5), equipment=("wheelchair", "accessible-bus"))
        with self.assertRaises(ValueError):
            manager.scan("PAX1", "FL100", "bus-to-lift", "s1", D(10, 16), equipment=("ambulift",))

    def test_pending_chain_reports_time_until_deadline(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        report = manager.status("PAX1", "FL100", D(9, 40))
        self.assertEqual(report.state, ChainState.ON_TRACK)
        self.assertEqual(report.responsible_party, "gate-agent")
        self.assertEqual(report.remaining_minutes, 40)
        self.assertEqual(manager.breach_report(D(9, 40)), ())

    def test_breach_report_ranks_most_urgent_first(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        manager.register(PrmJourney("PAX2", "FL200", (
            HandoverSegment(
                "gate-to-bus", "gate-agent", "bus-driver", frozenset(),
                Window(D(10, 0), D(10, 5)), D(10, 40),
            ),
        )))
        breaches = manager.breach_report(D(10, 25))
        self.assertEqual([r.passenger_id for r in breaches], ["PAX1", "PAX2"])
        self.assertEqual(breaches[0].state, ChainState.BROKEN)
        self.assertEqual(breaches[1].state, ChainState.OVERDUE)
        self.assertEqual(breaches[1].remaining_minutes, 15)

    def test_segment_deadline_must_cover_window(self):
        with self.assertRaises(ValueError):
            HandoverSegment("bad", "a", "b", frozenset(), Window(D(10), D(10, 10)), D(10, 5))

    def test_journey_segments_must_be_unique(self):
        seg = HandoverSegment("x", "a", "b", frozenset(), Window(D(10), D(10, 10)), D(10, 20))
        with self.assertRaises(ValueError):
            PrmJourney("P", "F", (seg, seg))

    def test_workflow_records_handover_event_once(self):
        manager = PrmHandoverManager()
        manager.register(journey())
        control = RampWorkflowControl()
        action = lambda: manager.scan(
            "PAX1", "FL100", "gate-to-bus", "s1", D(10, 5),
            equipment=("wheelchair", "accessible-bus"),
        )
        first = control.record_prm_handover("s1", "PAX1", D(10, 5), action)
        second = control.record_prm_handover("s1", "PAX1", D(10, 5), action)
        self.assertIs(first, second)
        self.assertEqual(control.stream.head, 1)
        self.assertTrue(first.result.accepted)
        self.assertEqual(manager.status("PAX1", "FL100", D(10, 6)).version, 2)


if __name__ == "__main__":
    unittest.main()
