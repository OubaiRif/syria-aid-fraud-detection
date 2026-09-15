"""
Arabic transliteration-variant name library.

Two dicts map a canonical Arabic name -> list of 2-4 Latin transliteration
variants. This is used two ways in the project:
  - Honest beneficiaries (Chunk 3): one variant is chosen ONCE per person
    and used consistently everywhere.
  - Duplicate-identity fraud (Chunk 4): the SAME underlying canonical name
    is reused across duplicate IDs, but a DIFFERENT variant is picked for
    each duplicate -- this is the data fingerprint fuzzy-matching rules
    (R2) are meant to catch.

normalize_name() is the (deliberately imperfect) canonicalizer that
detection rules will run names through.
"""

import re

FIRST_NAMES_MALE = {
    "محمد": ["Mohammed", "Muhammad", "Mohamad", "Mhd"],
    "أحمد": ["Ahmad", "Ahmed", "Ahmet"],
    "حسين": ["Hussein", "Husayn", "Hussain"],
    "حسن": ["Hassan", "Hasan"],
    "عبد الرحمن": ["Abdulrahman", "Abd al-Rahman", "Abdel Rahman", "Abdurrahman"],
    "عبد الله": ["Abdullah", "Abdallah", "Abd Allah"],
    "عمر": ["Omar", "Umar", "Omer"],
    "علي": ["Ali", "Aly", "Aliy"],
    "خالد": ["Khaled", "Khalid", "Khalild"],
    "يوسف": ["Youssef", "Yousef", "Yusuf", "Yousif"],
    "إبراهيم": ["Ibrahim", "Ebrahim", "Ibraheem"],
    "مصطفى": ["Mustafa", "Moustafa", "Mostafa"],
    "بلال": ["Bilal", "Belal"],
    "طارق": ["Tariq", "Tarek", "Tarik"],
    "زياد": ["Ziad", "Zeyad", "Ziyad"],
    "سامر": ["Samer", "Sameer"],
    "وائل": ["Wael", "Wa'el", "Wail"],
    "أنس": ["Anas", "Anass"],
    "فراس": ["Firas", "Feras"],
    "معاذ": ["Muath", "Mu'adh", "Moaz"],
    "بشار": ["Bashar", "Bachar"],
    "رامي": ["Rami", "Ramy"],
    "جمال": ["Jamal", "Gamal"],
    "نضال": ["Nidal", "Nithal"],
    "أيمن": ["Ayman", "Aiman"],
    "هيثم": ["Haitham", "Haytham"],
    "عماد": ["Emad", "Imad"],
    "فادي": ["Fadi", "Fady"],
    "سامي": ["Sami", "Samy"],
    "ماهر": ["Maher", "Mahir"],
}

FIRST_NAMES_FEMALE = {
    "فاطمة": ["Fatima", "Fatimah", "Fatma"],
    "عائشة": ["Aisha", "Aysha", "Aicha", "Ayesha"],
    "مريم": ["Maryam", "Mariam", "Mariyam"],
    "خديجة": ["Khadija", "Khadijah", "Khadiga"],
    "زينب": ["Zeinab", "Zainab", "Zaynab"],
    "سارة": ["Sara", "Sarah"],
    "نور": ["Nour", "Noor", "Nur"],
    "هدى": ["Huda", "Houda"],
    "سلمى": ["Salma", "Selma"],
    "رنا": ["Rana", "Rania"],
    "لينا": ["Lina", "Leena"],
    "غادة": ["Ghada", "Ghadah"],
    "منى": ["Mona", "Muna"],
    "أمل": ["Amal", "Amel"],
    "وفاء": ["Wafa", "Wafaa"],
}

FIRST_NAMES = {**FIRST_NAMES_MALE, **FIRST_NAMES_FEMALE}

FAMILY_NAMES = {
    "الخالدي": ["Al-Khalidi", "Alkhalidy", "Elkhalidi"],
    "الحلبي": ["Al-Halabi", "Alhalaby", "Halabi"],
    "الشامي": ["Al-Shami", "Alshamy", "Shami"],
    "العلي": ["Al-Ali", "Alaly", "Alali"],
    "الأحمد": ["Al-Ahmad", "Alahmad", "Elahmad"],
    "الحسين": ["Al-Hussein", "Alhussein", "Elhusayn"],
    "المصري": ["Al-Masri", "Almasry", "Elmasri"],
    "العبدالله": ["Al-Abdullah", "Alabdullah", "Elabdallah"],
    "النعيمي": ["Al-Naimi", "Alnuaimi", "Elnaimi"],
    "الجاسم": ["Al-Jassim", "Aljasem", "Eljasim"],
    "الزعبي": ["Al-Zoubi", "Alzoubi", "Elzo3bi"],
    "الحمصي": ["Al-Homsi", "Alhomsy", "Elhemsi"],
    "الإدلبي": ["Al-Idlibi", "Alidlibi", "Elidlibi"],
    "الحوراني": ["Al-Hourani", "Alhourani", "Elhorani"],
    "الشيخ": ["Al-Sheikh", "Alsheikh", "Elshiekh"],
    "الأسعد": ["Al-Asaad", "Alasaad", "Elasad"],
    "القاضي": ["Al-Qadi", "Alqady", "Elkadi"],
    "الدرويش": ["Al-Darwish", "Aldarwish", "Eldarwish"],
    "السيد": ["Al-Sayyed", "Alsayed", "Elsayed"],
    "الرفاعي": ["Al-Rifai", "Alrifay", "Elrefai"],
    "البكري": ["Al-Bakri", "Albakry", "Elbakri"],
    "الحموي": ["Al-Hamawi", "Alhamawy", "Elhamawi"],
    "الطويل": ["Al-Tawil", "Altaweel", "Eltawil"],
    "القصير": ["Al-Qasir", "Alqaseer", "Elkasir"],
    "الحاج": ["Al-Hajj", "Alhaj", "Elhag"],
    "المحمد": ["Al-Mohammad", "Almohammed", "Elmohamad"],
    "العمر": ["Al-Omar", "Alomar", "Elomar"],
    "الخطيب": ["Al-Khatib", "Alkhateeb", "Elkhatib"],
    "النجار": ["Al-Najjar", "Alnajar", "Elnaggar"],
    "الحداد": ["Al-Haddad", "Alhaddad", "Elhaddad"],
    "الصالح": ["Al-Saleh", "Alsaleh", "Elsaleh"],
    "الفارس": ["Al-Faris", "Alfares", "Elfaris"],
    "الكردي": ["Al-Kurdi", "Alkurdy", "Elkurdi"],
    "التركماني": ["Al-Turkmani", "Alturkmany", "Elturkmani"],
    "الشرقي": ["Al-Sharqi", "Alsharky", "Elsharqi"],
    "الغربي": ["Al-Gharbi", "Algharby", "Elgharbi"],
    "العباس": ["Al-Abbas", "Alabbas", "Elabbas"],
    "الحلاق": ["Al-Hallaq", "Alhallak", "Elhallaq"],
    "الخياط": ["Al-Khayyat", "Alkhayat", "Elkhayat"],
    "الشحادة": ["Al-Shahada", "Alshahadeh", "Elshahada"],
    "الجندي": ["Al-Jundi", "Aljondy", "Eljundi"],
    "السمان": ["Al-Samman", "Alsamman", "Elsaman"],
    "الزين": ["Al-Zein", "Alzain", "Elzein"],
    "النابلسي": ["Al-Nabulsi", "Alnabulsy", "Elnabulsi"],
    "الحلبوني": ["Al-Halbouni", "Alhalbouny", "Elhalbouni"],
    "الشوا": ["Al-Shawa", "Alshawwa", "Elshawa"],
    "القدسي": ["Al-Qudsi", "Alkudsi", "Elqudsi"],
    "الدمشقي": ["Al-Dimashqi", "Aldimashqy", "Eldimashqi"],
    "الحافظ": ["Al-Hafez", "Alhafedh", "Elhafez"],
    "السباعي": ["Al-Sibai", "Alsibaay", "Elsibai"],
    "العزب": ["Al-Azab", "Alazzab", "Elazab"],
    "الفتال": ["Al-Fattal", "Alfattal", "Elfattal"],
    "الحكيم": ["Al-Hakim", "Alhakeem", "Elhakim"],
    "المفتي": ["Al-Mufti", "Almufty", "Elmufti"],
    "الشعار": ["Al-Shaar", "Alshaar", "Elshaar"],
    "الترك": ["Al-Turk", "Alturk", "Elturk"],
    "الحلواني": ["Al-Halwani", "Alhalwany", "Elhalwani"],
    "الجابي": ["Al-Jabi", "Aljaby", "Eljabi"],
    "الخوري": ["Al-Khoury", "Alkhoury", "Elkhoury"],
    "الصباغ": ["Al-Sabbagh", "Alsabbagh", "Elsabbagh"],
}


def normalize_name(name: str) -> str:
    """
    Deliberately imperfect canonicalizer used by detection rules (R2).
    Lowercases, strips hyphens/spaces, drops leading al-/el- article,
    collapses doubled letters. It will NOT catch every variant pair --
    that imperfection is realistic and part of why R2 needs a similarity
    threshold rather than exact match.
    """
    if not name:
        return ""
    n = name.lower().strip()
    n = re.sub(r"[-\s']", "", n)
    n = re.sub(r"^(al|el)", "", n)
    n = re.sub(r"(.)\1+", r"\1", n)  # collapse doubled letters
    return n


def find_canonical(variant_text, name_dict):
    """
    Reverse-lookup: which canonical Arabic name does this Latin variant
    belong to? Used both by fraud injection (Chunk 4, to pick a DIFFERENT
    variant of the SAME canonical for a duplicate ID) and by detection
    (Chunk 5's R2, as a high-precision signal: two different variants of
    the same canonical name are a much stronger match than generic string
    similarity, which is dominated by coincidental shared surnames --
    real-world deduplication systems use exactly this kind of curated
    variant dictionary, not just fuzzy string matching).
    """
    for canonical, variants in name_dict.items():
        if variant_text in variants:
            return canonical
    return None
