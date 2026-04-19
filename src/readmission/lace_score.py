"""
Trevonti HBYS — LACE+ Readmission Risk Calculator
30-day hospital readmission prediction model.
"""
import math
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW       = "LOW"        # LACE 0-4  → %5 readmission risk
    MEDIUM    = "MEDIUM"     # LACE 5-9  → %15 readmission risk
    HIGH      = "HIGH"       # LACE 10-12 → %25 readmission risk
    VERY_HIGH = "VERY_HIGH"  # LACE 13+  → %35+ readmission risk


@dataclass
class LaceInput:
    length_of_stay_days:     int
    admission_type:          str   # "EMERGENCY" | "ELECTIVE"
    charlson_score:          int   # Comorbidity index (0-37)
    ed_visits_6months:       int   # ED visits in last 6 months


@dataclass
class LaceResult:
    lace_score:     int
    risk_level:     RiskLevel
    risk_score:     float          # 0.0 – 1.0
    top_factors:    list
    recommendation: str


def calculate_lace(inp: LaceInput) -> LaceResult:
    """
    LACE+ readmission risk score.
    L: Length of stay  (0-7 pts)
    A: Acuity          (0 or 3 pts)
    C: Comorbidity     (0-5 pts)
    E: ED visits       (0-4 pts)
    Max score: 19
    """
    l_score = _length_score(inp.length_of_stay_days)
    a_score = 3 if inp.admission_type == "EMERGENCY" else 0
    c_score = _charlson_score(inp.charlson_score)
    e_score = _ed_score(inp.ed_visits_6months)

    total = l_score + a_score + c_score + e_score

    return LaceResult(
        lace_score     = total,
        risk_level     = _classify_risk(total),
        risk_score     = _lace_to_probability(total),
        top_factors    = _explain_factors(l_score, a_score, c_score, e_score),
        recommendation = _get_recommendation(_classify_risk(total)),
    )


def _length_score(days: int) -> int:
    if days == 1:        return 1
    if days == 2:        return 2
    if days == 3:        return 3
    if 4 <= days <= 6:   return 4
    if 7 <= days <= 13:  return 5
    return 7  # 14+ days


def _charlson_score(score: int) -> int:
    if score == 0:  return 0
    if score == 1:  return 1
    if score == 2:  return 2
    if score == 3:  return 3
    return 5  # 4+


def _ed_score(visits: int) -> int:
    if visits == 0:  return 0
    if visits == 1:  return 1
    if visits == 2:  return 2
    if visits == 3:  return 3
    return 4  # 4+


def _lace_to_probability(lace: int) -> float:
    log_odds = -3.2 + 0.28 * lace
    return round(1 / (1 + math.exp(-log_odds)), 3)


def _classify_risk(lace: int) -> RiskLevel:
    if lace <= 4:   return RiskLevel.LOW
    if lace <= 9:   return RiskLevel.MEDIUM
    if lace <= 12:  return RiskLevel.HIGH
    return RiskLevel.VERY_HIGH


def _impact(score: int, max_score: int) -> str:
    ratio = score / max_score if max_score > 0 else 0
    if ratio >= 0.7:  return "HIGH"
    if ratio >= 0.4:  return "MEDIUM"
    return "LOW"


def _explain_factors(l, a, c, e) -> list:
    factors = [
        {"factor": "Yatış süresi",        "score": l, "max": 7, "impact": _impact(l, 7)},
        {"factor": "Acil kabul",           "score": a, "max": 3, "impact": _impact(a, 3)},
        {"factor": "Komorbidite (Charlson)","score": c, "max": 5, "impact": _impact(c, 5)},
        {"factor": "Acil başvuru (6 ay)",  "score": e, "max": 4, "impact": _impact(e, 4)},
    ]
    return sorted(factors, key=lambda x: x["score"], reverse=True)


def _get_recommendation(risk: RiskLevel) -> str:
    recs = {
        RiskLevel.LOW:
            "Standart taburcu planı uygulanabilir.",
        RiskLevel.MEDIUM:
            "Taburcu öncesi hasta eğitimi ve 7 günlük takip randevusu planlayın.",
        RiskLevel.HIGH:
            "Discharge planner dahil edin. Ev bakım servisi değerlendirin.",
        RiskLevel.VERY_HIGH:
            "Yoğun taburcu planlaması gerekli. Sosyal hizmet ve ev bakım zorunlu.",
    }
    return recs[risk]
