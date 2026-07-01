"""
data/generate_data.py
Generates synthetic labeled document text for 6 document classes.
Run: python -m src.data.generate_data
"""
import random
import json
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ── Name/place corpora ─────────────────────────────────────────────────────
FIRST_NAMES = [
    "Ahmed", "Mohammed", "Sara", "Fatima", "Omar", "Layla", "Khalid", "Noor",
    "Ravi", "Priya", "James", "Emily", "Wei", "Lin", "Carlos", "Maria",
    "Bhushan", "Ananya", "Arjun", "Divya", "Rahul", "Sneha",
]
LAST_NAMES = [
    "Al-Rashid", "Hassan", "Sharma", "Patel", "Smith", "Johnson", "Zhang",
    "Garcia", "Kumar", "Singh", "Chen", "Williams", "Brown", "Davis", "Iyer",
]
NATIONALITIES = [
    "Indian", "Saudi Arabian", "Emirati", "Pakistani", "Egyptian",
    "Filipino", "American", "British", "German", "Chinese",
]
COUNTRIES = [
    "India", "Saudi Arabia", "UAE", "Pakistan", "Egypt",
    "Philippines", "USA", "UK", "Germany", "China",
]
BANKS = [
    "State Bank of India", "HDFC Bank", "ICICI Bank", "Emirates NBD",
    "Al Rajhi Bank", "Citibank", "HSBC", "Deutsche Bank", "Bank of America",
]
UTILITIES = [
    "DEWA", "ADDC", "Saudi Electricity Company", "BESCOM",
    "Tata Power", "Con Edison", "EDF Energy",
]
CITIES = [
    "Dubai", "Mumbai", "Delhi", "Riyadh", "Cairo",
    "Manila", "London", "New York", "Berlin", "Shanghai", "Bangalore",
]
STREETS = [
    "Al Wasl Road", "MG Road", "King Fahd Road", "Corniche Street",
    "Main Street", "Park Avenue", "Oxford Street", "Nehru Nagar", "Anna Salai",
]


# ── Helpers ────────────────────────────────────────────────────────────────
def rname():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"

def rdate(start=1960, end=2005):
    s = datetime(start, 1, 1)
    return s + timedelta(days=random.randint(0, (datetime(end, 12, 31) - s).days))

def rfuture(years=5):
    return datetime.now() + timedelta(days=random.randint(30, years * 365))

def rid(n=9):
    return "".join(str(random.randint(0, 9)) for _ in range(n))

def raddr():
    return f"{random.randint(1, 999)} {random.choice(STREETS)}, {random.choice(CITIES)}"

def ramount(lo=500, hi=50_000):
    return f"{random.uniform(lo, hi):.2f}"


# ── Per-class generators ───────────────────────────────────────────────────
def gen_passport():
    name = rname(); dob = rdate(); exp = rfuture(10)
    nat = random.choice(NATIONALITIES); cty = random.choice(COUNTRIES)
    pno = f"{random.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rid(8)}"
    sex = random.choice(["M", "F"])
    mrz1 = f"P<{cty[:3].upper()}{''.join(name.upper().split())[:39]:<39}"
    mrz2 = f"{pno}{rid(1)}{cty[:3].upper()}{dob.strftime('%y%m%d')}{sex}{exp.strftime('%y%m%d')}"
    text = (
        f"PASSPORT\nRepublic of {cty}\nINTERNATIONAL TRAVEL DOCUMENT\n\n"
        f"Surname / Nom: {name.split()[-1].upper()}\n"
        f"Given Names / Prénoms: {name.split()[0].upper()}\n"
        f"Nationality / Nationalité: {nat}\n"
        f"Date of Birth / Date de naissance: {dob.strftime('%d %b %Y')}\n"
        f"Sex / Sexe: {sex}\n"
        f"Place of Birth: {random.choice(CITIES)}\n"
        f"Date of Issue: {(datetime.now()-timedelta(days=random.randint(30,365))).strftime('%d %b %Y')}\n"
        f"Date of Expiry / Date d'expiration: {exp.strftime('%d %b %Y')}\n"
        f"Passport No. / No. du passeport: {pno}\n"
        f"Personal No.: {rid(14)}\n\n"
        f"Machine Readable Zone:\n{mrz1}\n{mrz2}\n"
    )
    entities = {
        "Name": name, "Nationality": nat,
        "Date of Birth": dob.strftime("%d %b %Y"),
        "Expiry Date": exp.strftime("%d %b %Y"),
        "Passport Number": pno, "Sex": sex,
    }
    return text, entities


def gen_national_id():
    name = rname(); dob = rdate(); exp = rfuture(5)
    nat = random.choice(NATIONALITIES); addr = raddr()
    id_no = rid(15)
    text = (
        f"NATIONAL IDENTITY CARD\nGOVERNMENT ISSUED IDENTIFICATION\n\n"
        f"ID Number: {id_no}\n"
        f"Full Name: {name}\n"
        f"Date of Birth: {dob.strftime('%d/%m/%Y')}\n"
        f"Nationality: {nat}\n"
        f"Gender: {random.choice(['Male','Female'])}\n"
        f"Blood Type: {random.choice(['A+','B+','O+','AB+','A-','O-'])}\n"
        f"Address: {addr}\n"
        f"Issue Date: {(datetime.now()-timedelta(days=random.randint(30,1800))).strftime('%d/%m/%Y')}\n"
        f"Expiry Date: {exp.strftime('%d/%m/%Y')}\n"
        f"Issuing Authority: Ministry of Interior\n"
        f"Card Serial: {rid(8)}\n\n"
        f"This card is the property of the issuing government.\n"
    )
    entities = {
        "Name": name, "ID Number": id_no, "Nationality": nat,
        "Date of Birth": dob.strftime("%d/%m/%Y"),
        "Expiry Date": exp.strftime("%d/%m/%Y"), "Address": addr,
    }
    return text, entities


def gen_commercial_registration():
    company = (
        f"{random.choice(LAST_NAMES)} "
        f"{random.choice(['Trading','Industries','Holdings','Group','Enterprises','Solutions','Technologies'])} LLC"
    )
    reg_no = f"CR-{rid(7)}"
    issue = datetime.now() - timedelta(days=random.randint(365, 3650))
    exp = issue + timedelta(days=365 * random.randint(1, 5))
    owner = rname(); addr = raddr()
    activity = random.choice([
        "General Trading", "Import & Export", "IT Services",
        "Consulting Services", "Construction", "Food & Beverages",
        "Real Estate", "Healthcare Services",
    ])
    capital = f"USD {random.randint(50, 5000) * 1000:,}"
    text = (
        f"CERTIFICATE OF COMMERCIAL REGISTRATION\nMinistry of Commerce and Industry\n\n"
        f"Registration Number: {reg_no}\n"
        f"Company Name: {company}\n"
        f"Legal Form: Limited Liability Company (LLC)\n"
        f"Business Activity: {activity}\n"
        f"Registered Address: {addr}\n"
        f"Owner / Manager: {owner}\n"
        f"Authorized Capital: {capital}\n"
        f"Date of Establishment: {issue.strftime('%d %B %Y')}\n"
        f"Registration Date: {issue.strftime('%d %B %Y')}\n"
        f"Expiry Date: {exp.strftime('%d %B %Y')}\n"
        f"License Number: LIC-{rid(8)}\n"
        f"Tax Registration Number: TRN{rid(12)}\n\n"
        f"Issued by: Registrar of Companies\nOfficial Seal and Signature\n"
    )
    entities = {
        "Company Name": company, "Registration Number": reg_no,
        "Owner": owner, "Business Activity": activity,
        "Expiry Date": exp.strftime("%d %B %Y"),
        "Address": addr, "Capital": capital,
    }
    return text, entities


def gen_bank_statement():
    name = rname(); bank = random.choice(BANKS)
    acc = f"****{rid(4)}"; iban = f"AE{rid(23)}"
    stmt_date = datetime.now() - timedelta(days=random.randint(1, 30))
    opening = float(ramount())
    txns = []
    bal = opening
    for _ in range(random.randint(8, 15)):
        t = stmt_date - timedelta(days=random.randint(1, 30))
        amt = random.uniform(50, 5000)
        if random.random() > 0.4:
            bal += amt
            txns.append(f"{t.strftime('%d/%m/%Y')}  {'Salary/Transfer/Deposit':<30}  CR  {amt:>10.2f}  {bal:>12.2f}")
        else:
            bal = max(bal - amt, abs(bal - amt) * 0.1)
            txns.append(f"{t.strftime('%d/%m/%Y')}  {'Payment/Withdrawal/POS':<30}  DR  {amt:>10.2f}  {bal:>12.2f}")
    text = (
        f"{bank}\nACCOUNT STATEMENT\n\n"
        f"Account Holder: {name}\n"
        f"Account Number: {acc}\n"
        f"IBAN: {iban}\n"
        f"Account Type: Current Account\n"
        f"Currency: USD\n"
        f"Statement Period: {(stmt_date-timedelta(days=30)).strftime('%d/%m/%Y')} to {stmt_date.strftime('%d/%m/%Y')}\n"
        f"Statement Date: {stmt_date.strftime('%d/%m/%Y')}\n"
        f"Branch: {random.choice(CITIES)} Main Branch\n\n"
        f"Opening Balance: {opening:,.2f}\n"
        f"DATE        DESCRIPTION                        TYPE  AMOUNT        BALANCE\n"
        f"{'-'*80}\n"
        f"{chr(10).join(txns)}\n"
        f"{'-'*80}\n"
        f"Closing Balance: {bal:,.2f}\n\n"
        f"This statement is computer generated.\n"
    )
    entities = {
        "Account Holder": name, "Account Number": acc,
        "Bank": bank, "IBAN": iban,
        "Statement Date": stmt_date.strftime("%d/%m/%Y"),
        "Closing Balance": f"{bal:,.2f}",
    }
    return text, entities


def gen_utility_bill():
    name = rname(); util = random.choice(UTILITIES); addr = raddr()
    acc = f"UTL{rid(9)}"; meter = f"MTR{rid(8)}"
    bill_date = datetime.now() - timedelta(days=random.randint(1, 30))
    due_date = bill_date + timedelta(days=21)
    units = random.randint(200, 2000)
    amount = units * random.uniform(0.08, 0.25)
    text = (
        f"{util}\nUTILITY BILL / TAX INVOICE\n\n"
        f"Customer Name: {name}\n"
        f"Service Address: {addr}\n"
        f"Account Number: {acc}\n"
        f"Meter Number: {meter}\n"
        f"Customer Reference: {rid(10)}\n\n"
        f"Bill Date: {bill_date.strftime('%d %B %Y')}\n"
        f"Due Date: {due_date.strftime('%d %B %Y')}\n"
        f"Billing Period: {(bill_date-timedelta(days=30)).strftime('%d %B %Y')} to {bill_date.strftime('%d %B %Y')}\n\n"
        f"METER READINGS\n"
        f"Previous Reading: {random.randint(10000,99000)}\n"
        f"Current Reading:  {random.randint(10000,99000)}\n"
        f"Units Consumed:   {units} kWh\n\n"
        f"CHARGES BREAKDOWN\n"
        f"Energy Charges:        {amount*0.75:.2f}\n"
        f"Distribution Charges:  {amount*0.15:.2f}\n"
        f"Government Levy:       {amount*0.05:.2f}\n"
        f"VAT (5%):             {amount*0.05:.2f}\n"
        f"{'-'*40}\n"
        f"TOTAL AMOUNT DUE:     {amount:.2f}\n\n"
        f"Pay online at www.{util.lower().replace(' ','')}.com\n"
    )
    entities = {
        "Customer Name": name, "Service Address": addr,
        "Account Number": acc, "Bill Date": bill_date.strftime("%d %B %Y"),
        "Due Date": due_date.strftime("%d %B %Y"),
        "Amount Due": f"{amount:.2f}", "Utility Provider": util,
    }
    return text, entities


def gen_invoice():
    vendor = f"{random.choice(LAST_NAMES)} {random.choice(['Corp','Ltd','Inc','Co','Services'])}."
    client = rname()
    inv_no = f"INV-{rid(6)}"
    inv_date = datetime.now() - timedelta(days=random.randint(1, 90))
    due_date = inv_date + timedelta(days=30)
    items, subtotal = [], 0
    for _ in range(random.randint(2, 5)):
        desc = random.choice([
            "Professional Services", "Software License", "Consulting Fee",
            "Hardware Supply", "Maintenance", "Training", "Support",
        ])
        qty = random.randint(1, 10)
        unit = random.uniform(100, 2000)
        total = qty * unit; subtotal += total
        items.append(f"{desc:<30}  {qty:>3}  {unit:>10.2f}  {total:>12.2f}")
    tax = subtotal * 0.05; total = subtotal + tax
    text = (
        f"COMMERCIAL INVOICE\n\n"
        f"From: {vendor}\n"
        f"       {raddr()}\n\n"
        f"To:   {client}\n"
        f"      {raddr()}\n\n"
        f"Invoice Number: {inv_no}\n"
        f"Invoice Date:   {inv_date.strftime('%d %B %Y')}\n"
        f"Due Date:       {due_date.strftime('%d %B %Y')}\n"
        f"Payment Terms:  Net 30 days\n\n"
        f"DESCRIPTION                        QTY    UNIT PRICE      TOTAL\n"
        f"{'-'*65}\n"
        f"{chr(10).join(items)}\n"
        f"{'-'*65}\n"
        f"                                          Subtotal:  {subtotal:>12.2f}\n"
        f"                                          VAT (5%):  {tax:>12.2f}\n"
        f"                                          TOTAL:     {total:>12.2f}\n\n"
        f"Bank: {random.choice(BANKS)}\n"
        f"Account: {rid(12)}\n"
        f"IBAN: AE{rid(23)}\n\n"
        f"Thank you for your business.\n"
    )
    entities = {
        "Vendor": vendor, "Client": client, "Invoice Number": inv_no,
        "Invoice Date": inv_date.strftime("%d %B %Y"),
        "Due Date": due_date.strftime("%d %B %Y"),
        "Total Amount": f"{total:.2f}",
    }
    return text, entities


GENERATORS = {
    "Passport":                gen_passport,
    "National ID":             gen_national_id,
    "Commercial Registration": gen_commercial_registration,
    "Bank Statement":          gen_bank_statement,
    "Utility Bill":            gen_utility_bill,
    "Invoice":                 gen_invoice,
}


def generate_dataset(n_per_class: int = 200, out_path: str = "data/dataset.json"):
    data = []
    for label, fn in GENERATORS.items():
        for _ in range(n_per_class):
            text, entities = fn()
            data.append({"text": text, "label": label, "entities": entities})
    random.shuffle(data)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated {len(data)} samples → {out_path}")
    for label in GENERATORS:
        print(f"  {label}: {sum(1 for d in data if d['label']==label)}")
    return data


if __name__ == "__main__":
    generate_dataset()
