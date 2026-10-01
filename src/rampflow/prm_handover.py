from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Iterable, Mapping

from .timeutil import Window, utc


class SegmentState(str, Enum):
    PENDING = "pending"
    DUE = "due"
    OVERDUE = "overdue"
    BROKEN = "broken"
    COMPLETED = "completed"


class ChainState(str, Enum):
    ON_TRACK = "on_track"
    OVERDUE = "overdue"
    BROKEN = "broken"
    COMPLETED = "completed"


@dataclass(frozen=True)
class HandoverSegment:
    """One custody transfer: from_party hands the passenger to to_party.

    Custody only moves when the receiving scan confirms it; until then
    accountability stays with from_party.  required_equipment lists the
    equipment kinds that must be confirmed at the scan (wheelchair,
    accessible bus, ambulift, ...).  recover_by is the hard remediation
    deadline: after it the segment can no longer connect and the chain
    is broken.
    """

    segment_id: str
    from_party: str
    to_party: str
    required_equipment: frozenset[str]
    window: Window
    recover_by: datetime

    def __post_init__(self):
        object.__setattr__(self, "recover_by", utc(self.recover_by))
        if not self.segment_id or not self.from_party or not self.to_party:
            raise ValueError("segment id and both parties required")
        if self.recover_by < self.window.end:
            raise ValueError("recovery deadline precedes planned window")


@dataclass(frozen=True)
class PrmJourney:
    passenger_id: str
    flight_id: str
    segments: tuple[HandoverSegment, ...]
    completed: tuple[str, ...] = ()
    completions: Mapping[str, datetime] = field(default_factory=dict)
    version: int = 1

    def __post_init__(self):
        if not self.passenger_id or not self.flight_id:
            raise ValueError("passenger and flight required")
        if not self.segments:
            raise ValueError("journey needs at least one segment")
        ids = [s.segment_id for s in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate segment id")
        starts = [s.window.start for s in self.segments]
        if starts != sorted(starts):
            raise ValueError("segments must be chronological")
        if set(self.completed) != set(self.completions):
            raise ValueError("completion record mismatch")
        if any(c not in set(ids) for c in self.completed):
            raise ValueError("unknown completed segment")


@dataclass(frozen=True)
class ScanReceipt:
    scan_id: str
    passenger_id: str
    flight_id: str
    segment_id: str
    accepted: bool
    code: str
    reason: str
    recorded_at: datetime
    version: int
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))


@dataclass(frozen=True)
class SegmentView:
    segment_id: str
    state: SegmentState
    from_party: str
    to_party: str
    required_equipment: tuple[str, ...]
    window: Window
    recover_by: datetime
    remaining_minutes: int | None
    completed_at: datetime | None


@dataclass(frozen=True)
class ChainReport:
    passenger_id: str
    flight_id: str
    state: ChainState
    current_segment: str | None
    responsible_party: str | None
    expected_receiver: str | None
    required_equipment: tuple[str, ...]
    recover_by: datetime | None
    remaining_minutes: int | None
    segments: tuple[SegmentView, ...]
    version: int


class PrmHandoverManager:
    """Scan-confirmed custody chain for passengers with reduced mobility.

    A verbal handover leaves no record; here custody only moves when the
    receiving party scans.  Scans are idempotent by scan_id: a repeated
    scan returns the original receipt and is never counted as a second
    completion.  At any moment, status() and breach_report() show who
    currently holds the passenger and the hard deadline by which an
    unconnected segment can still be remediated.
    """

    def __init__(self):
        self.journeys: dict[str, PrmJourney] = {}
        self._scans: dict[str, ScanReceipt] = {}

    def _key(self, passenger_id: str, flight_id: str) -> str:
        return f"{passenger_id}:{flight_id}"

    def register(self, journey: PrmJourney) -> None:
        key = self._key(journey.passenger_id, journey.flight_id)
        if key in self.journeys:
            raise ValueError("duplicate journey")
        if journey.completed:
            raise ValueError("journey must register uncompleted")
        self.journeys[key] = journey

    def _current(self, journey: PrmJourney) -> HandoverSegment | None:
        done = set(journey.completed)
        for segment in journey.segments:
            if segment.segment_id not in done:
                return segment
        return None

    def scan(
        self,
        passenger_id: str,
        flight_id: str,
        segment_id: str,
        scan_id: str,
        at: datetime,
        equipment: Iterable[str] = (),
    ) -> ScanReceipt:
        at = utc(at)
        previous = self._scans.get(scan_id)
        if previous is not None:
            identity = (previous.passenger_id, previous.flight_id, previous.segment_id)
            if identity != (passenger_id, flight_id, segment_id):
                raise ValueError("scan id reused for a different handover")
            return previous
        journey = self.journeys[self._key(passenger_id, flight_id)]
        segment = next((s for s in journey.segments if s.segment_id == segment_id), None)
        if segment is None:
            raise KeyError(segment_id)
        if isinstance(equipment, str):
            equipment = (equipment,)
        provided = frozenset(equipment)

        def rejected(code: str, reason: str, details: Mapping[str, object] | None = None) -> ScanReceipt:
            return ScanReceipt(
                scan_id, passenger_id, flight_id, segment_id,
                False, code, reason, at, journey.version, details or {},
            )

        if segment_id in journey.completed:
            return rejected("duplicate-segment", "segment already completed; rescan not counted")
        current = self._current(journey)
        if current is None or current.segment_id != segment_id:
            return rejected(
                "out-of-order",
                "earlier segment still unconfirmed",
                {"awaiting": current.segment_id if current else None},
            )
        missing = tuple(sorted(set(segment.required_equipment) - set(provided)))
        if missing:
            return rejected(
                "equipment-missing",
                "required equipment not confirmed at handover",
                {"missing": missing},
            )
        if at > segment.recover_by:
            return rejected(
                "recovery-expired",
                "remediation deadline passed; chain requires escalation",
                {"recover_by": segment.recover_by},
            )
        updated = replace(
            journey,
            completed=journey.completed + (segment_id,),
            completions={**journey.completions, segment_id: at},
            version=journey.version + 1,
        )
        self.journeys[self._key(passenger_id, flight_id)] = updated
        receipt = ScanReceipt(
            scan_id, passenger_id, flight_id, segment_id,
            True, "ok", "handover confirmed", at, updated.version,
            {"equipment": tuple(sorted(provided))},
        )
        self._scans[scan_id] = receipt
        return receipt

    def _segment_view(self, journey: PrmJourney, segment: HandoverSegment, at: datetime) -> SegmentView:
        completed_at = journey.completions.get(segment.segment_id)
        equipment = tuple(sorted(segment.required_equipment))
        if completed_at is not None:
            return SegmentView(
                segment.segment_id, SegmentState.COMPLETED,
                segment.from_party, segment.to_party, equipment,
                segment.window, segment.recover_by, None, completed_at,
            )
        if at < segment.window.start:
            state = SegmentState.PENDING
        elif at < segment.window.end:
            state = SegmentState.DUE
        elif at <= segment.recover_by:
            state = SegmentState.OVERDUE
        else:
            state = SegmentState.BROKEN
        remaining = int((segment.recover_by - at).total_seconds() // 60)
        return SegmentView(
            segment.segment_id, state,
            segment.from_party, segment.to_party, equipment,
            segment.window, segment.recover_by, remaining, None,
        )

    def _report(self, journey: PrmJourney, at: datetime) -> ChainReport:
        views = tuple(self._segment_view(journey, s, at) for s in journey.segments)
        current = self._current(journey)
        if current is None:
            return ChainReport(
                journey.passenger_id, journey.flight_id, ChainState.COMPLETED,
                None, journey.segments[-1].to_party, None, (), None, None,
                views, journey.version,
            )
        view = next(v for v in views if v.segment_id == current.segment_id)
        state = {
            SegmentState.PENDING: ChainState.ON_TRACK,
            SegmentState.DUE: ChainState.ON_TRACK,
            SegmentState.OVERDUE: ChainState.OVERDUE,
            SegmentState.BROKEN: ChainState.BROKEN,
        }[view.state]
        return ChainReport(
            journey.passenger_id, journey.flight_id, state,
            current.segment_id, current.from_party, current.to_party,
            view.required_equipment, current.recover_by, view.remaining_minutes,
            views, journey.version,
        )

    def status(self, passenger_id: str, flight_id: str, at: datetime) -> ChainReport:
        at = utc(at)
        return self._report(self.journeys[self._key(passenger_id, flight_id)], at)

    def breach_report(self, at: datetime) -> tuple[ChainReport, ...]:
        at = utc(at)
        reports = [self._report(journey, at) for journey in self.journeys.values()]
        breaches = [r for r in reports if r.state in (ChainState.OVERDUE, ChainState.BROKEN)]
        return tuple(sorted(breaches, key=lambda r: (r.remaining_minutes, r.passenger_id, r.flight_id)))
