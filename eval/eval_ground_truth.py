"""Ground truth dataset for retrieval evaluation.

Each entry contains:
- query: A natural language question
- relevant_ids: Set of parent_ids that are relevant for this query
- description: Brief explanation of why these docs are relevant
"""

from __future__ import annotations

GROUND_TRUTH: list[dict] = [
    {
        "query": "What is the profit rate for NUST Asaan Account?",
        "relevant_ids": {"RATE_001", "RATE_004", "NAA_001"},
        "description": "Asaan Account rate info from rate sheet",
    },
    {
        "query": "How do I open a Little Champs Account for my child?",
        "relevant_ids": {"LCA_001", "LCA_002", "LCA_003", "LCA_004", "LCA_005"},
        "description": "Little Champs Account eligibility and features",
    },
    {
        "query": "What is the auto finance markup rate?",
        "relevant_ids": {"NUST4Car_001"},
        "description": "Auto Assist Finance rate information",
    },
    {
        "query": "What are the NUST Hunarmand Finance eligibility requirements?",
        "relevant_ids": {"NHF_001", "NHF_002", "NHF_003"},
        "description": "NHF eligibility criteria",
    },
    {
        "query": "What are the charges for late payments on Hunarmand Finance?",
        "relevant_ids": {"NHF_011"},
        "description": "Late payment charges",
    },
    {
        "query": "How do I send money from abroad to Pakistan using NUST Bank?",
        "relevant_ids": {
            "HOME REMITTANCE_001",
            "HOME REMITTANCE_002",
            "HOME REMITTANCE_021",
        },
        "description": "Home Remittance service information",
    },
    {
        "query": "What is the limit for cash over the counter remittance?",
        "relevant_ids": {"HOME REMITTANCE_004"},
        "description": "CoC transaction limit",
    },
    {
        "query": "Are there any charges for home remittance service?",
        "relevant_ids": {"HOME REMITTANCE_005"},
        "description": "Remittance charges",
    },
    {
        "query": "How can I purchase a Bancassurance policy?",
        "relevant_ids": {"Nust Life_001", "EFU Life_001", "Jubilee Life _001"},
        "description": "Bancassurance policy purchase eligibility",
    },
    {
        "query": "What insurance plans are available through NUST Bank?",
        "relevant_ids": {
            "Nust Life_001",
            "Nust Life_002",
            "Nust Life_003",
            "EFU Life_001",
            "EFU Life_002",
            "EFU Life_003",
            "Jubilee Life _001",
            "Jubilee Life _002",
            "Jubilee Life _003",
        },
        "description": "All Bancassurance plans",
    },
    {
        "query": "What is the minimum premium for insurance plans?",
        "relevant_ids": {
            "Nust Life_002",
            "Nust Life_003",
            "EFU Life_002",
            "EFU Life_003",
            "Jubilee Life _002",
            "Jubilee Life _003",
        },
        "description": "Insurance minimum premium amounts",
    },
    {
        "query": "What is the Senior Citizen Waqaar Account profit rate?",
        "relevant_ids": {
            "RATE_015",
            "RATE_019",
            "RATE_021",
            "RATE_023",
            "RATE_025",
            "RATE_026",
            "RATE_027",
            "RATE_030",
        },
        "description": "Waqaar Account rates",
    },
    {
        "query": "What are the term deposit rates for different tenors?",
        "relevant_ids": {
            "RATE_005",
            "RATE_006",
            "RATE_007",
            "RATE_009",
            "RATE_010",
            "RATE_011",
            "RATE_012",
            "RATE_014",
            "RATE_016",
            "RATE_018",
            "RATE_020",
            "RATE_022",
            "RATE_024",
            "RATE_028",
            "RATE_029",
            "RATE_031",
            "RATE_032",
            "RATE_033",
            "RATE_034",
            "RATE_035",
        },
        "description": "Various term deposit rates",
    },
    {
        "query": "How do I change my funds transfer limit in the mobile app?",
        "relevant_ids": {"FT_001", "FT_002"},
        "description": "Funds transfer limit change",
    },
    {
        "query": "How can I add a beneficiary for funds transfer?",
        "relevant_ids": {"FT_003"},
        "description": "Adding beneficiaries",
    },
    {
        "query": "What is the daily transfer limit?",
        "relevant_ids": {"FT_001", "FT_004"},
        "description": "Transfer limits",
    },
    {
        "query": "How do I reset my mobile banking password?",
        "relevant_ids": {"AF_002"},
        "description": "Password reset",
    },
    {
        "query": "Can I use the app while traveling abroad?",
        "relevant_ids": {"AF_001"},
        "description": "International app usage",
    },
    {
        "query": "How do I pay utility bills through the app?",
        "relevant_ids": {"AF_005"},
        "description": "Bill payment service",
    },
    {
        "query": "What is the markup rate for auto finance?",
        "relevant_ids": {"NUST4Car_001", "NUST4Car_002"},
        "description": "Auto finance markup",
    },
    {
        "query": "How many years is the maximum tenure for auto finance?",
        "relevant_ids": {"NUST4Car_001"},
        "description": "Auto finance tenure",
    },
    {
        "query": "What documents are needed for auto finance?",
        "relevant_ids": {"NUST4Car_001", "NUST4Car_002"},
        "description": "Auto finance requirements",
    },
    {
        "query": "Is there a free look period for insurance?",
        "relevant_ids": {"Nust Life_003", "EFU Life_004", "Jubilee Life _003"},
        "description": "Insurance free look period",
    },
    {
        "query": "What happens if I cancel my insurance policy early?",
        "relevant_ids": {"Nust Life_003", "EFU Life_004", "Jubilee Life _003"},
        "description": "Policy cancellation terms",
    },
    {
        "query": "Can I receive remittance without a bank account?",
        "relevant_ids": {"HOME REMITTANCE_007"},
        "description": "Remittance without account",
    },
    {
        "query": "How can I track my remittance status?",
        "relevant_ids": {"HOME REMITTANCE_011"},
        "description": "Remittance tracking",
    },
    {
        "query": "What is RAAST and how do I use it?",
        "relevant_ids": {"FT_005", "FT_006"},
        "description": "RAAST payment system",
    },
    {
        "query": "How do I enable international transactions on my debit card?",
        "relevant_ids": {"FT_004"},
        "description": "Card international activation",
    },
    {
        "query": "What are the banking hours for remittance services?",
        "relevant_ids": {"HOME REMITTANCE_012"},
        "description": "Banking hours for remittance",
    },
]
