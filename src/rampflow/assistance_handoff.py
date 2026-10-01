from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Iterable

from .models import Decision
from .timeutil import utc


class HandoffState(str, Enum):
    WAITING = "waiting"
    OVERDUE = "overdue"
    EXPIRED = "expired"
    BLOCKED = "blocked"
    COMPLETE = "complete"


@dataclass(frozen=True)
class AssistanceEquipment:
    equipment_id: str
    kind: str
    available: bool = True

    def __post_init__(self) -> None:
        if not self.equipment_id or not self.kind:
            raise ValueError("equipment id and kind are required")


@dataclass(frozen=True)
class AssistanceLeg:
    leg_id: str
    from_party: str
    to_party: str
    location: str
    required_equipment: frozenset[str]
    scheduled_by: datetime
    remedy_by: datetime

    def __post_init__(self) -> None:
        scheduled_by = utc(self.scheduled_by)
        remedy_by = utc(self.remedy_by)
        if not self.leg_id or not self.from_party or not self.to_party or not self.location:
            raise ValueError("leg identity and parties are required")
        if not self.required_equipment:
            raise ValueError("at least one equipment kind is required")
        if remedy_by < scheduled_by:
            raise ValueError("remedy cutoff must not precede scheduled handoff")
        object.__setattr__(self, "required_equipment", frozenset(self.required_equipment))
        object.__setattr__(self, "scheduled_by", scheduled_by)
        object.__setattr__(self, "remedy_by", remedy_by)


@dataclass(frozen=True)
class AssistanceJourney:
    journey_id: str
    passenger_id: str
    flight_id: str
    legs: tuple[AssistanceLeg, ...]

    def __post_init__(self) -> None:
        legs = tuple(self.legs)
        if not self.journey_id or not self.passenger_id or not self.flight_id:
            raise ValueError("journey identity is required")
        if not legs:
            raise ValueError("assistance journey requires at least one leg")
        seen: set[str] = set()
        previous: AssistanceLeg | None = None
        for leg in legs:
            if leg.leg_id in seen:
                raise ValueError(f"duplicate leg {leg.leg_id}")
            if previous is not None:
                if leg.from_party != previous.to_party:
                    raise ValueError(
                        f"leg {leg.leg_id} breaks responsibility chain: "
                        f"{previous.to_party} must hand to {leg.from_party}"
                    )
                if leg.scheduled_by < previous.scheduled_by or leg.remedy_by < previous.remedy_by:
                    raise ValueError(f"leg {leg.leg_id} has an earlier cutoff than its predecessor")
            seen.add(leg.leg_id)
            previous = leg
        object.__setattr__(self, "legs", legs)


@dataclass(frozen=True)
class HandoffScan:
    scan_id: str
    journey_id: str
    leg_id: str
    scanned_at: datetime
    receiving_party: str
    equipment_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.scan_id or not self.journey_id or not self.leg_id:
            raise ValueError("scan identity is required")
        if not self.receiving_party:
            raise ValueError("receiving party is required")
        object.__setattr__(self, "scanned_at", utc(self.scanned_at))
        object.__setattr__(self, "equipment_ids", tuple(sorted(set(self.equipment_ids))))


@dataclass(frozen=True)
class HandoffReceipt:
    scan_id: str
    journey_id: str
    leg_id: str
    accepted_at: datetime
    receiving_party: str
    equipment_ids: tuple[str, ...]
    sequence: int


@dataclass(frozen=True)
class LegStatus:
    index: int
    leg_id: str
    from_party: str
    to_party: str
    location: str
    state: HandoffState
    responsible_party: str | None
    awaiting_party: str | None
    required_equipment: tuple[str, ...]
    ready_equipment: tuple[str, ...]
    missing_equipment: tuple[str, ...]
    used_equipment: tuple[str, ...]
    scheduled_by: datetime
    remedy_by: datetime
    remaining_minutes: int | None
    completed_at: datetime | None = None


@dataclass(frozen=True)
class JourneyStatus:
    journey_id: str
    passenger_id: str
    flight_id: str
    state: HandoffState
    current_leg_id: str | None
    current_responsible_party: str | None
    awaiting_party: str | None
    remedy_by: datetime | None
    remaining_minutes: int | None
    completed_legs: int
    total_legs: int
    missing_equipment: tuple[str, ...]
    legs: tuple[LegStatus, ...]

    @property
    def connected(self) -> bool:
        return self.state is HandoffState.COMPLETE


class AssistanceHandoffManager:
    """Chain-of-custody tracker for reduced-mobility passenger handoffs."""

    def __init__(self, equipment: Iterable[AssistanceEquipment] = ()):
        self.equipment: dict[str, AssistanceEquipment] = {}
        for item in equipment:
            if item.equipment_id in self.equipment:
                raise ValueError(f"duplicate equipment {item.equipment_id}")
            self.equipment[item.equipment_id] = item
        self.journeys: dict[str, AssistanceJourney] = {}
        self._receipts: dict[str, dict[str, HandoffReceipt]] = {}
        self._scan_ids: dict[str, HandoffReceipt] = {}
        self._decisions: dict[str, Decision] = {}
        self._lock = RLock()

    def register(self, journey: AssistanceJourney) -> None:
        with self._lock:
            if journey.journey_id in self.journeys:
                raise ValueError(f"journey {journey.journey_id} already registered")
            self.journeys[journey.journey_id] = journey
            self._receipts[journey.journey_id] = {}

    def record_scan(self, scan: HandoffScan) -> Decision:
        with self._lock:
            duplicated = self._scan_ids.get(scan.scan_id)
            if duplicated is not None:
                if not self._same_scan_identity(scan, duplicated):
                    raise ValueError(f"scan {scan.scan_id} reused with different handoff identity")
                return self._decisions[scan.scan_id]

            journey = self.journeys.get(scan.journey_id)
            if journey is None:
                raise KeyError(scan.journey_id)
            leg_index = self._leg_index(journey, scan.leg_id)
            leg = journey.legs[leg_index]
            receipts = self._receipts[journey.journey_id]

            if scan.receiving_party != leg.to_party:
                return Decision(
                    False,
                    "wrong-party",
                    "only the receiving party can accept custody",
                    {"expected_party": leg.to_party, "scanned_party": scan.receiving_party},
                )

            gap = self._first_open_index(journey, receipts)
            if gap < leg_index:
                gap_leg = journey.legs[gap]
                return Decision(
                    False,
                    "chain-gap",
                    "an earlier handoff is not connected",
                    self._gap_details(gap_leg, scan.scanned_at),
                )

            previous = receipts.get(journey.legs[leg_index - 1].leg_id) if leg_index > 0 else None
            if previous is not None and scan.scanned_at < previous.accepted_at:
                return Decision(
                    False,
                    "scan-order",
                    "handoff scan is earlier than the previous acceptance",
                    {"previous_scan_id": previous.scan_id, "previous_at": previous.accepted_at},
                )

            existing = receipts.get(leg.leg_id)
            if existing is not None:
                return Decision(
                    True,
                    "duplicate",
                    "handoff already complete; repeat scan was not counted",
                    {
                        "counted": False,
                        "original_scan_id": existing.scan_id,
                        "journey_id": journey.journey_id,
                        "leg_id": leg.leg_id,
                        "completed_legs": len(receipts),
                        "total_legs": len(journey.legs),
                    },
                )

            if scan.scanned_at > leg.remedy_by:
                return Decision(
                    False,
                    "remedy-expired",
                    "the handoff can no longer be completed in time",
                    self._gap_details(leg, scan.scanned_at),
                )

            equipment_decision = self._check_equipment(leg, scan)
            if equipment_decision is not None:
                return equipment_decision

            receipt = HandoffReceipt(
                scan.scan_id,
                journey.journey_id,
                leg.leg_id,
                scan.scanned_at,
                scan.receiving_party,
                scan.equipment_ids,
                leg_index + 1,
            )
            receipts[leg.leg_id] = receipt
            self._scan_ids[scan.scan_id] = receipt

            details = {
                "counted": True,
                "journey_id": journey.journey_id,
                "leg_id": leg.leg_id,
                "completed_at": receipt.accepted_at,
                "equipment_used": receipt.equipment_ids,
                "completed_legs": len(receipts),
                "total_legs": len(journey.legs),
            }
            if leg_index + 1 < len(journey.legs):
                next_leg = journey.legs[leg_index + 1]
                details.update(
                    {
                        "current_responsible_party": leg.to_party,
                        "awaiting_party": next_leg.to_party,
                        "next_leg_id": next_leg.leg_id,
                        "remedy_by": next_leg.remedy_by,
                        "remaining_minutes": self._minutes_until(next_leg.remedy_by, scan.scanned_at),
                    }
                )
            else:
                details.update(
                    {
                        "current_responsible_party": leg.to_party,
                        "awaiting_party": None,
                        "next_leg_id": None,
                        "remedy_by": None,
                        "remaining_minutes": None,
                    }
                )

            decision = Decision(True, "ok", "PRM handoff accepted", details)
            self._decisions[scan.scan_id] = decision
            return decision

    def status(self, journey_id: str, at: datetime) -> JourneyStatus:
        with self._lock:
            journey = self.journeys.get(journey_id)
            if journey is None:
                raise KeyError(journey_id)
            now = utc(at)
            receipts = self._receipts[journey_id]
            first_open = self._first_open_index(journey, receipts)
            leg_statuses: list[LegStatus] = []

            for index, leg in enumerate(journey.legs):
                receipt = receipts.get(leg.leg_id)
                required = tuple(sorted(leg.required_equipment))
                if receipt is not None:
                    leg_statuses.append(
                        LegStatus(
                            index,
                            leg.leg_id,
                            leg.from_party,
                            leg.to_party,
                            leg.location,
                            HandoffState.COMPLETE,
                            leg.to_party,
                            None,
                            required,
                            receipt.equipment_ids,
                            (),
                            receipt.equipment_ids,
                            leg.scheduled_by,
                            leg.remedy_by,
                            None,
                            receipt.accepted_at,
                        )
                    )
                    continue

                ready = tuple(
                    sorted(
                        item.equipment_id
                        for item in self.equipment.values()
                        if item.available and item.kind in leg.required_equipment
                    )
                )
                ready_kinds = {self.equipment[item_id].kind for item_id in ready}
                missing = tuple(sorted(leg.required_equipment - ready_kinds))

                if index != first_open:
                    leg_statuses.append(
                        LegStatus(
                            index,
                            leg.leg_id,
                            leg.from_party,
                            leg.to_party,
                            leg.location,
                            HandoffState.BLOCKED,
                            None,
                            None,
                            required,
                            ready,
                            missing,
                            (),
                            leg.scheduled_by,
                            leg.remedy_by,
                            None,
                        )
                    )
                    continue

                if now > leg.remedy_by:
                    state = HandoffState.EXPIRED
                elif now > leg.scheduled_by:
                    state = HandoffState.OVERDUE
                else:
                    state = HandoffState.WAITING
                leg_statuses.append(
                    LegStatus(
                        index,
                        leg.leg_id,
                        leg.from_party,
                        leg.to_party,
                        leg.location,
                        state,
                        leg.from_party,
                        leg.to_party,
                        required,
                        ready,
                        missing,
                        (),
                        leg.scheduled_by,
                        leg.remedy_by,
                        self._minutes_until(leg.remedy_by, now),
                    )
                )

            if first_open >= len(journey.legs):
                final_party = journey.legs[-1].to_party
                return JourneyStatus(
                    journey.journey_id,
                    journey.passenger_id,
                    journey.flight_id,
                    HandoffState.COMPLETE,
                    None,
                    final_party,
                    None,
                    None,
                    None,
                    len(journey.legs),
                    len(journey.legs),
                    (),
                    tuple(leg_statuses),
                )

            current = leg_statuses[first_open]
            return JourneyStatus(
                journey.journey_id,
                journey.passenger_id,
                journey.flight_id,
                current.state,
                current.leg_id,
                current.responsible_party,
                current.awaiting_party,
                current.remedy_by,
                current.remaining_minutes,
                first_open,
                len(journey.legs),
                current.missing_equipment,
                tuple(leg_statuses),
            )

    def _first_open_index(self, journey: AssistanceJourney, receipts: dict[str, HandoffReceipt]) -> int:
        for index, leg in enumerate(journey.legs):
            if leg.leg_id not in receipts:
                return index
        return len(journey.legs)

    def _check_equipment(self, leg: AssistanceLeg, scan: HandoffScan) -> Decision | None:
        used: list[AssistanceEquipment] = []
        unknown: list[str] = []
        unavailable: list[str] = []
        for equipment_id in scan.equipment_ids:
            item = self.equipment.get(equipment_id)
            if item is None:
                unknown.append(equipment_id)
            elif not item.available:
                unavailable.append(equipment_id)
            else:
                used.append(item)

        if unknown:
            return Decision(
                False,
                "unknown-equipment",
                "scan references equipment that is not registered",
                {"unknown_equipment": tuple(sorted(unknown)), "leg_id": leg.leg_id},
            )
        if unavailable:
            return Decision(
                False,
                "equipment-unavailable",
                "required equipment is not serviceable",
                {"unavailable_equipment": tuple(sorted(unavailable)), "leg_id": leg.leg_id},
            )

        provided_kinds = {item.kind for item in used}
        missing = tuple(sorted(leg.required_equipment - provided_kinds))
        if missing:
            return Decision(
                False,
                "equipment-missing",
                "the receiving party cannot accept without all required equipment",
                {
                    "leg_id": leg.leg_id,
                    "required_equipment": tuple(sorted(leg.required_equipment)),
                    "missing_equipment": missing,
                },
            )
        return None

    def _gap_details(self, leg: AssistanceLeg, at: datetime) -> dict[str, object]:
        now = utc(at)
        if now > leg.remedy_by:
            state = HandoffState.EXPIRED
        elif now > leg.scheduled_by:
            state = HandoffState.OVERDUE
        else:
            state = HandoffState.WAITING
        ready_equipment = tuple(
            sorted(
                item.equipment_id
                for item in self.equipment.values()
                if item.available and item.kind in leg.required_equipment
            )
        )
        ready_kinds = {self.equipment[item_id].kind for item_id in ready_equipment}
        missing_equipment = tuple(sorted(leg.required_equipment - ready_kinds))
        return {
            "current_leg_id": leg.leg_id,
            "state": state,
            "responsible_party": leg.from_party,
            "awaiting_party": leg.to_party,
            "location": leg.location,
            "required_equipment": tuple(sorted(leg.required_equipment)),
            "ready_equipment": ready_equipment,
            "missing_equipment": missing_equipment,
            "remedy_by": leg.remedy_by,
            "remaining_minutes": self._minutes_until(leg.remedy_by, now),
        }

    @staticmethod
    def _leg_index(journey: AssistanceJourney, leg_id: str) -> int:
        for index, leg in enumerate(journey.legs):
            if leg.leg_id == leg_id:
                return index
        raise KeyError(leg_id)

    @staticmethod
    def _same_scan_identity(scan: HandoffScan, receipt: HandoffReceipt) -> bool:
        return (
            scan.journey_id == receipt.journey_id
            and scan.leg_id == receipt.leg_id
            and scan.scanned_at == receipt.accepted_at
            and scan.receiving_party == receipt.receiving_party
            and scan.equipment_ids == receipt.equipment_ids
        )

    def register_equipment(self, item: AssistanceEquipment) -> None:
        with self._lock:
            if item.equipment_id in self.equipment:
                raise ValueError(f"duplicate equipment {item.equipment_id}")
            self.equipment[item.equipment_id] = item

    def add_equipment(self, item: AssistanceEquipment) -> None:
        self.register_equipment(item)

    def register_journey(self, journey: AssistanceJourney) -> None:
        self.register(journey)

    def accept_scan(self, scan: HandoffScan) -> Decision:
        return self.record_scan(scan)

    def handoff_status(self, journey_id: str, at: datetime) -> JourneyStatus:
        return self.status(journey_id, at)

    @staticmethod
    def _minutes_until(deadline: datetime, now: datetime) -> int:
        return int((deadline - utc(now)).total_seconds() // 60)


class MobilityHandoffManager(AssistanceHandoffManager):
    pass


# PRM is the standard aviation abbreviation for passengers with reduced mobility.
PRMHandoffManager = MobilityHandoffManager
PRMHandoverManager = PRMHandoffManager
HandoffManager = MobilityHandoffManager
HandoffEquipment = AssistanceEquipment
HandoffLeg = AssistanceLeg
HandoffJourney = AssistanceJourney
PRMHandoff = HandoffScan
PRMHandover = HandoffScan
