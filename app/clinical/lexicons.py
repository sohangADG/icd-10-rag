"""Small, generic clinical-language lexicons for the rule-based extractor.

These describe how clinicians *write* (negation/uncertainty cues, section headers, standard
abbreviations, laterality words...). They contain no classification content.
"""

import re

SECTION_HEADERS: dict[str, str] = {
    "chief complaint": "chief_complaint",
    "cc": "chief_complaint",
    "history of present illness": "hpi",
    "hpi": "hpi",
    "past medical history": "past_history",
    "pmh": "past_history",
    "past history": "past_history",
    "family history": "family_history",
    "fhx": "family_history",
    "fh": "family_history",
    "social history": "social_history",
    "shx": "social_history",
    "assessment": "assessment",
    "impression": "assessment",
    "assessment and plan": "assessment",
    "diagnosis": "assessment",
    "diagnoses": "assessment",
    "discharge diagnosis": "assessment",
    "discharge diagnoses": "assessment",
    "final diagnosis": "assessment",
    "problem list": "assessment",
    "plan": "plan",
    "procedure": "procedures",
    "procedures": "procedures",
    "medications": "medications",
    "meds": "medications",
    "allergies": "allergies",
    "review of systems": "ros",
    "ros": "ros",
    "physical exam": "exam",
    "examination": "exam",
    "findings": "findings",
    "admin": "administrative",
    "administrative": "administrative",
    "billing": "administrative",
}
# Sections whose content is not a list of conditions to code.
SKIPPED_SECTIONS = {"plan", "medications", "allergies", "administrative"}

# (regex, status) — checked in order; first match wins. All anchored to clause start or end.
RULED_OUT_PATTERNS = [
    re.compile(r"\b(?:was|were|has been|been)?\s*ruled\s+out\b", re.I),
    re.compile(r"^\s*(?:excluded)\b", re.I),
    # "AKI excluded", "Diabetes was excluded" (cue after the condition)
    re.compile(r"\s*\b(?:is|was|were|has been|have been)?\s*excluded\s*$", re.I),
]
# A negative finding after a short lead-in: "Workup negative for X", "Echo shows no evidence
# of X" (at most three lead-in words; the condition follows the cue).
NEGATION_INFIX = re.compile(
    r"^(?:[\w/-]+\s+){1,3}?(negative for|no evidence of|no signs? of|without evidence of)\s+",
    re.I,
)
# Grammatical subject before a cue ("Patient denies X", "Pt has no X").
SUBJECT_PREFIX = re.compile(
    r"^(?:(?:the\s+)?(?:patient|pt|he|she)\s+)?(?:(?:has|have|had)\s+(?=(?:no|not)\b))?", re.I
)
# A short label before a colon that is not a known section header ("Dx: CHF",
# "CXR: no consolidation", "Copied from prior note: ..."): not part of the concept.
LABEL_PREFIX = re.compile(r"^[A-Za-z][\w/'-]*(?:\s+[\w/'-]+){0,3}\s*:\s+(?=\S)")
NEGATION_PREFIXES = [
    "no evidence of",
    "no signs of",
    "no sign of",
    "no history of",
    "negative for",
    "absence of",
    "free of",
    "denies",
    "denied",
    "without",
    "not",
    "no",
]
SUSPECTED_PREFIXES = [
    "rule out",
    "r/o",
    "suspected",
    "suspect",
    "suspicious for",
    "probable",
    "probably",
    "likely",
    "presumed",
    "presumptive",
    "concern for",
    "consistent with",
    "compatible with",
]
UNCERTAIN_PREFIXES = [
    "cannot exclude",
    "cannot rule out",
    "can't rule out",
    "possible",
    "possibly",
    "questionable",
    "query",
    "?",
]
FAMILY_PREFIXES = [
    "family history of",
    "family hx of",
    "fhx of",
    "fh of",
    "fhx",
]
FAMILY_MEMBER = re.compile(
    r"^(?:his|her|the patient's|patient's)?\s*(?:mother|father|sister|brother|parent|parents|"
    r"grandmother|grandfather|aunt|uncle|sibling|siblings|son|daughter)\s+"
    r"(?:had|has|with|died of|diagnosed with)\s+",
    re.I,
)
HISTORY_PREFIXES = [
    "personal history of",
    "past history of",
    "history of",
    "hx of",
    "h/o",
    "status post",
    "s/p",
]
HISTORY_WORDS = {"former", "previous", "prior", "resolved", "remote", "ex", "quit"}
NEGATION_TERMINATORS = re.compile(r"\b(?:but|however|although|except|aside from)\b", re.I)
# A clause that starts a new statement ("... and the patient has X") ends any scoped cue.
NEW_STATEMENT = re.compile(
    r"^(?:(?:the\s+)?(?:patient|pt)|he|she|they|has|have|had|reports?|presents?|"
    r"diagnosed|is|was|shows?)\b",
    re.I,
)
# Words that only qualify a condition. A clause made only of them ("left" in "left and right
# knee pain") qualifies the next clause instead of being a concept of its own.
QUALIFIER_WORDS = frozenset(
    {
        "left",
        "right",
        "bilateral",
        "both",
        "mild",
        "moderate",
        "severe",
        "acute",
        "chronic",
        "subacute",
        "lt",
        "rt",
    }
)

# Filler at the start of a clause that carries no clinical meaning.
FILLER_PREFIXES = [
    "patient reports",
    "reports",
    "reported",
    "patient presents with",
    "presents with",
    "presenting with",
    "patient has",
    "patient with",
    "pt with",
    "pt has",
    "the patient has",
    "has",
    "had",
    "diagnosed with",
    "diagnosis of",
    "known",
    "complains of",
    "c/o",
    "admitted with",
    "admitted for",
    "seen for",
    "here for",
    "findings of",
    "evidence of",
    "ongoing",
    "new",
    "current",
]

ABBREVIATIONS: dict[str, str] = {
    "HTN": "hypertension",
    "DM": "diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "T2DM": "type 2 diabetes mellitus",
    "CKD": "chronic kidney disease",
    "AKI": "acute kidney injury",
    "CHF": "congestive heart failure",
    "HF": "heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "MI": "myocardial infarction",
    "CAD": "coronary artery disease",
    "UTI": "urinary tract infection",
    "URTI": "upper respiratory tract infection",
    "SOB": "shortness of breath",
    "AF": "atrial fibrillation",
    "AFIB": "atrial fibrillation",
    "PNA": "pneumonia",
    "CVA": "cerebrovascular accident",
    "TIA": "transient ischaemic attack",
    "GERD": "gastro-oesophageal reflux disease",
    "OA": "osteoarthritis",
    "RA": "rheumatoid arthritis",
    "DVT": "deep vein thrombosis",
    "HLD": "hyperlipidaemia",
    "OSA": "obstructive sleep apnoea",
    "BPH": "benign prostatic hyperplasia",
}

LATERALITY = {
    "bilateral": "bilateral",
    "both": "bilateral",
    "left": "left",
    "lt": "left",
    "right": "right",
    "rt": "right",
}
SEVERITY = {"mild": "mild", "moderate": "moderate", "severe": "severe"}
ACUITY_PATTERNS = [
    (re.compile(r"\bacute[\s-]+on[\s-]+chronic\b", re.I), "acute on chronic"),
    (re.compile(r"\bsubacute\b", re.I), "subacute"),
    (re.compile(r"\bacute\b", re.I), "acute"),
    (re.compile(r"\b(?:chronic|long[\s-]standing)\b", re.I), "chronic"),
]
ENCOUNTER_PATTERNS = [
    (re.compile(r"\binitial (?:encounter|visit)\b", re.I), "initial"),
    (re.compile(r"\bsubsequent (?:encounter|visit)\b", re.I), "subsequent"),
    (re.compile(r"\bsequela[e]?\b", re.I), "sequela"),
    (re.compile(r"\bfollow[\s-]?up\b", re.I), "follow-up"),
]
SUBTYPE_PATTERN = re.compile(r"\btype\s*(1|2|i{1,2}|one|two)\b", re.I)
STAGE_PATTERN = re.compile(r"\bstage\s*([0-9]+[a-b]?|i{1,3}v?|iv|v)\b", re.I)
COMPLICATION_PATTERN = re.compile(
    r"\b(?:with|complicated by)\s+(?!no\b|out\b)(.+?)(?=$|,|;|\bdue to\b|\bsecondary to\b)", re.I
)
ABSENCE_PATTERN = re.compile(
    r"\b(?:without|no)\s+(complications?|kidney involvement|organ involvement|"
    r"[a-z]+ (?:complication|involvement))\b|\buncomplicated\b",
    re.I,
)
# "from" is deliberately not a cue: "copied from prior note", "discharged from hospital".
CAUSE_PATTERN = re.compile(r"\b(?:due to|secondary to|caused by|because of)\s+(.+?)(?=$|,|;)", re.I)
ANATOMY = {
    "lung": "lung",
    "lungs": "lung",
    "pulmonary": "lung",
    "kidney": "kidney",
    "kidneys": "kidney",
    "renal": "kidney",
    "heart": "heart",
    "cardiac": "heart",
    "liver": "liver",
    "hepatic": "liver",
    "brain": "brain",
    "airway": "airway",
    "airways": "airway",
    "bronchial": "airway",
    "chest": "chest",
    "abdomen": "abdomen",
    "abdominal": "abdomen",
    "bladder": "bladder",
    "skin": "skin",
    "eye": "eye",
    "ear": "ear",
    "knee": "knee",
    "hip": "hip",
    "ankle": "ankle",
    "wrist": "wrist",
    "shoulder": "shoulder",
    "elbow": "elbow",
    "foot": "foot",
    "hand": "hand",
    "arm": "arm",
    "leg": "leg",
    "spine": "spine",
    "lobe": "lobe",
    "lobes": "lobe",
    "legs": "leg",
    "arms": "arm",
    "knees": "knee",
    "hips": "hip",
    "ankles": "ankle",
    "wrists": "wrist",
    "shoulders": "shoulder",
    "elbows": "elbow",
    "feet": "foot",
    "hands": "hand",
    "eyes": "eye",
    "ears": "ear",
    "breast": "breast",
    "breasts": "breast",
    "side": "side",
    "sides": "side",
    "limb": "limb",
    "limbs": "limb",
    "extremity": "limb",
    "extremities": "limb",
}
# Laterality words count only when an anatomical site follows within this many words
# ("left lower lobe", "left knee"), never on their own ("patient left the clinic").
LATERALITY_WINDOW = 3
SYMPTOMS = {
    "tired",
    "tiredness",
    "lethargy",
    "chills",
    "pain",
    "ache",
    "fever",
    "cough",
    "nausea",
    "vomiting",
    "dizziness",
    "fatigue",
    "headache",
    "dyspnoea",
    "dyspnea",
    "wheeze",
    "wheezing",
    "rash",
    "swelling",
    "bleeding",
    "diarrhoea",
    "diarrhea",
    "constipation",
    "itching",
    "weakness",
    "malaise",
    "palpitations",
}
PROCEDURE_PATTERN = re.compile(
    r"^(?:underwent|performed|insertion of|removal of|placement of|repair of|biopsy)\b|"
    r"\b\w+(?:ectomy|otomy|ostomy|plasty|scopy)\b",
    re.I,
)
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
        "this",
        "that",
        "patient",
        "pt",
        "today",
        "noted",
        "seen",
    ]
)
