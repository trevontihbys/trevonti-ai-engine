"""
Trevonti HBYS — Sepsis Early Warning System
Hybrid qSOFA + NEWS2 clinical model.
Triggers automatically on vital sign recording.
"""
from dataclasses import dataclass


@dataclass
class VitalSigns:
    respiratory_rate:      int
    systolic_bp:           int
    altered_mental_status: bool
    heart_rate:            int
    temperature:           float
    spo2:                  float
    news2_score:           int


@dataclass
class SepsisRisk:
    risk_score:    float
    risk_level:    str       # LOW | MEDIUM | HIGH | CRITICAL
    qsofa_score:   int
    news2_score:   int
    alerts:        list
    sepsis_bundle: list


def calculate_sepsis_risk(vitals: VitalSigns) -> SepsisRisk:
    """
    Hybrid qSOFA + NEWS2 sepsis early warning.

    qSOFA criteria (1 point each, max 3):
      - Respiratory rate ≥ 22/min
      - Systolic BP ≤ 100 mmHg
      - Altered mental status (GCS < 15)

    Combined with NEWS2 score for higher sensitivity.
    """
    qsofa = 0
    alerts = []

    if vitals.respiratory_rate >= 22:
        qsofa += 1
        alerts.append(
            f"Solunum hızı yüksek: {vitals.respiratory_rate}/dk (eşik: ≥22)"
        )

    if vitals.systolic_bp <= 100:
        qsofa += 1
        alerts.append(
            f"Sistolik KB düşük: {vitals.systolic_bp} mmHg (eşik: ≤100)"
        )

    if vitals.altered_mental_status:
        qsofa += 1
        alerts.append("Bilinç değişikliği tespit edildi (GCS < 15)")

    # Ek uyarılar
    if vitals.spo2 < 92:
        alerts.append(f"SpO₂ kritik düşük: %{vitals.spo2} (normal: ≥%95)")

    if vitals.temperature >= 38.3 or vitals.temperature <= 36.0:
        alerts.append(f"Anormal ateş: {vitals.temperature}°C")

    if vitals.heart_rate >= 130:
        alerts.append(f"Taşikardi: {vitals.heart_rate}/dk")

    # Kombine risk skoru (qSOFA %60 + NEWS2 %40)
    news2_component = min(vitals.news2_score / 20.0, 1.0)
    qsofa_component = qsofa / 3.0
    combined = round((qsofa_component * 0.6) + (news2_component * 0.4), 3)

    # Risk sınıflandırma
    if qsofa >= 2 or vitals.news2_score >= 7:
        level = "CRITICAL"
    elif qsofa >= 1 or vitals.news2_score >= 5:
        level = "HIGH"
    elif vitals.news2_score >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"

    return SepsisRisk(
        risk_score    = combined,
        risk_level    = level,
        qsofa_score   = qsofa,
        news2_score   = vitals.news2_score,
        alerts        = alerts,
        sepsis_bundle = _get_sepsis_bundle(level),
    )


def _get_sepsis_bundle(level: str) -> list:
    """
    Surviving Sepsis Campaign bundle önerileri.
    """
    if level == "CRITICAL":
        return [
            "Kan kültürü al (antibiyotik öncesi, 2 set)",
            "Geniş spektrumlu antibiyotik başla (1 saat içinde)",
            "Laktik asit ölç (hedef: <2 mmol/L)",
            "IV sıvı resüsitasyonu: 30 mL/kg kristaloid (3 saat içinde)",
            "Saatlik idrar çıkışı takibi (hedef: >0.5 mL/kg/saat)",
            "YBÜ konsültasyonu — acil",
            "Vazopressör hazırlığı (MAP <65 mmHg ise)",
        ]
    if level == "HIGH":
        return [
            "Laktik asit ve kan kültürü",
            "IV sıvı başla",
            "Sorumlu hekim bilgilendir — acil",
            "15 dakikada bir vital takip",
            "Antibiyotik başlama değerlendirmesi",
        ]
    if level == "MEDIUM":
        return [
            "Vital takip sıklığını artır (30 dk)",
            "Klinik durumu yeniden değerlendir",
            "Yeterli IV erişim sağla",
        ]
    return [
        "Rutin vital takip devam et",
        "Klinik deteriorasyon için izle",
    ]
