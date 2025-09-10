# ml/augment_faq.py
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "faq_train.json"

# ONE paraphrase per singleton answer (answers MUST match exactly)
NEW_ITEMS = [
  {"q":"Can I book a visit through the Health Portal or by calling extension 3000?","a":"Yes. Use the Health Portal or call Clinic Reception at extension 3000."},
  {"q":"I’m a new patient—where do I register before my first visit?","a":"Complete the New Patient form in the Health Portal and bring photo ID and insurance card to your first visit."},
  {"q":"What documents should I bring to my consultation?","a":"Bring photo ID, insurance card, a list of current medications, and any relevant medical reports."},
  {"q":"How early should I arrive ahead of my appointment time?","a":"Please arrive 15 minutes early to allow time for check-in and paperwork."},
  {"q":"Where do I check in for MRI scans in the hospital?","a":"MRI is performed in Radiology (Building D, Ground Floor). Please check in at the Radiology desk."},
  {"q":"Which building and floor is Pediatrics located on?","a":"Pediatrics is in Building C, Level 1. Follow the green signs."},
  {"q":"Where can I find the Orthopedics department?","a":"Orthopedics is in Building B, Level 3."},
  {"q":"What’s the process to request a medication refill?","a":"Request a refill through the Health Portal or ask your pharmacy to send an e-prescription. Allow up to 2 business days."},
  {"q":"Can you e-prescribe to my preferred pharmacy if I provide the details?","a":"Yes. We support e-prescribing to your preferred pharmacy—provide the pharmacy name and location at check-in."},
  {"q":"Do refills for controlled medicines require an in-person visit?","a":"Yes. Controlled substances generally require an in-person evaluation per policy and local regulations."},
  {"q":"Do I need to fast for my blood test, and for how many hours?","a":"Some tests require fasting for 8–12 hours. Drink water only unless your clinician advised otherwise."},
  {"q":"How long do lab results take to appear in the Health Portal?","a":"Most lab results are posted to the Health Portal within 2–3 business days. Your clinician will contact you if follow-up is needed."},
  {"q":"Do I need a referral or prior authorization for CT or MRI?","a":"Most advanced imaging requires a clinician referral, and your insurer may require prior authorization."},
  {"q":"What’s the prep for an abdominal ultrasound—do I need to avoid food?","a":"Do not eat for 6 hours before the exam. You may drink small sips of water and take essential medications."},
  {"q":"Can I bring previous scans on USB, or can the hospital request them?","a":"Yes. Please bring prior images on CD/USB or provide the facility details so we can request them."},
  {"q":"Will I need to pay a copay at check-in, and how is it determined?","a":"If your plan requires a copay, it is collected at check-in. Amounts vary—check your insurance card."},
  {"q":"How do I request a price estimate from the Billing Office?","a":"Contact the Billing Office (Building A, Room 210) or submit a request via the Billing section of the Health Portal."},
  {"q":"Where exactly is Medical Records and what are the hours?","a":"Medical Records is in Building A, Room 115. Hours: Mon-Fri 09:00-17:00."},
  {"q":"Where should hospital visitors park their cars?","a":"Visitors should use Parking Lot P1 near the Main Gate."},
  {"q":"Do you validate parking for clinic appointments, and for how long?","a":"Yes. Clinic visits qualify for parking validation up to 2 hours. Validate at the clinic reception desk."},
  {"q":"Is there a weekday shuttle from the Main Gate to the Hospital Entrance?","a":"Yes. The shuttle runs Mon-Fri 07:00-19:00 every 15 minutes between the Main Gate and the Hospital Entrance."},
  {"q":"Are wheelchairs available in the Main Lobby if I need one?","a":"Yes. Wheelchairs are available at the Main Lobby—ask the Security Desk or a volunteer."},
  {"q":"Can I request an interpreter and how do I arrange it?","a":"Yes. Professional interpreters are available at no cost. Please request at least 48 hours in advance via the Health Portal or call extension 3000."},
  {"q":"Who should I contact to arrange accessibility accommodations for my visit?","a":"Contact Accessibility Services in Building A, Room 120, or note your needs in the Health Portal before your visit."},
  {"q":"Are service animals permitted in hospital areas?","a":"Yes. Service animals are welcome. Pets are not permitted except for approved therapy programs."},
  {"q":"Where is Lost & Found located in the hospital?","a":"Lost and Found is at the Security Desk in the Main Lobby."},
  {"q":"How can I request a doctor’s note or medical certificate?","a":"Ask at Clinic Reception or via the Health Portal. Processing typically takes 2 business days."},
  {"q":"When and where are the regular blood donation drives held?","a":"Yes. A blood donation drive is held on the first Wednesday of each month, 09:00-14:00, in the Auditorium (Building F)."}
]

def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    before = len(data)
    # de-duplicate on (q,a)
    seen = {(x["q"].strip(), x["a"].strip()) for x in data}
    added = 0
    for it in NEW_ITEMS:
        key = (it["q"].strip(), it["a"].strip())
        if key not in seen:
            data.append(it); seen.add(key); added += 1
    DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Added {added} items. FAQ size: {before} -> {len(data)}")

if __name__ == "__main__":
    main()
