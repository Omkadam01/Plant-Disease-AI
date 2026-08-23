"""Grape disease information for the desktop application."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiseaseInfo:
    """Structured disease information for UI display."""

    name: str
    description: str
    symptoms: list[str]
    treatment: list[str]
    prevention: list[str]
    severity: str  # low | medium | high


DISEASE_DATABASE: dict[str, DiseaseInfo] = {
    "Black Rot": DiseaseInfo(
        name="Black Rot",
        description=(
            "Black rot is a serious fungal disease caused by Guignardia bidwellii. "
            "It affects grape leaves, shoots, and fruit, causing significant yield "
            "losses in vineyards worldwide. Early detection and management are critical."
        ),
        symptoms=[
            "Circular tan spots with dark reddish-brown borders on leaves",
            "Small black fruiting bodies (pycnidia) appearing in lesion centers",
            "Shriveled, mummified berries turning black",
            "V-shaped brown lesions on leaf margins spreading toward veins",
            "Premature leaf drop in severe infections",
        ],
        treatment=[
            "Remove and destroy infected leaves, mummified berries, and canes",
            "Apply fungicides containing myclobutanil or captan during bloom period",
            "Improve canopy airflow through proper pruning and leaf removal",
            "Apply protective fungicide sprays at 10-14 day intervals during wet periods",
            "Consult local agricultural extension for region-specific fungicide programs",
        ],
        prevention=[
            "Plant resistant grape varieties when available",
            "Maintain open canopy architecture for air circulation",
            "Remove mummified fruit and infected debris from vineyard floor",
            "Apply dormant season copper or lime sulfur sprays",
            "Monitor weather conditions and apply preventive fungicides before rain",
        ],
        severity="high",
    ),
    "Esca (Black Measles)": DiseaseInfo(
        name="Esca (Black Measles)",
        description=(
            "Esca is a complex trunk disease caused by multiple wood-decay fungi "
            "(Phaeoacremonium, Phaeomoniella, Fomitiporia). It is one of the most "
            "destructive grapevine diseases, causing gradual vine decline over several years."
        ),
        symptoms=[
            "Tiger-stripe pattern on leaves (interveinal chlorosis and necrosis)",
            "Sudden wilting and dieback of shoots and leaves (apoplexy)",
            "Dark streaking visible in cross-sections of infected wood",
            "Small black spots (measles) on berries near harvest",
            "Reduced vigor and uneven ripening across the vine",
        ],
        treatment=[
            "Surgically remove infected wood using trunk renewal techniques",
            "Apply wound protectants after pruning cuts",
            "Remove and destroy severely affected vines to prevent spread",
            "Trunk injection therapies may be available in some regions",
            "There is no fully effective chemical cure; focus on vine recovery practices",
        ],
        prevention=[
            "Use clean, sanitized pruning tools between vines",
            "Avoid large pruning wounds; prune during dry weather",
            "Delay pruning in cold climates to promote wound healing",
            "Select planting material certified free of trunk pathogens",
            "Maintain balanced vine nutrition and avoid water stress",
        ],
        severity="high",
    ),
    "Leaf Blight (Isariopsis Leaf Spot)": DiseaseInfo(
        name="Leaf Blight (Isariopsis Leaf Spot)",
        description=(
            "Leaf blight, also known as Isariopsis leaf spot, is a fungal disease "
            "caused by Pseudocercosporella vitis. It primarily affects grape leaves "
            "and can reduce photosynthetic capacity when infections are severe."
        ),
        symptoms=[
            "Irregular brown to dark brown spots on leaf surfaces",
            "Spots may coalesce forming larger necrotic patches",
            "Yellow halos surrounding some lesions",
            "Premature defoliation in heavily infected canopies",
            "Lesions more common on older leaves in lower canopy",
        ],
        treatment=[
            "Remove heavily infected leaves from the vineyard",
            "Apply appropriate foliar fungicides if infection is widespread",
            "Improve canopy ventilation through shoot positioning and leaf removal",
            "Avoid overhead irrigation that keeps foliage wet for extended periods",
            "Monitor and treat adjacent vines if disease is spreading",
        ],
        prevention=[
            "Practice good canopy management for airflow and light penetration",
            "Avoid excessive nitrogen fertilization that promotes dense foliage",
            "Clean up fallen leaf debris at end of growing season",
            "Use preventive fungicide programs in historically affected blocks",
            "Scout regularly during humid, warm growing conditions",
        ],
        severity="medium",
    ),
    "Healthy": DiseaseInfo(
        name="Healthy",
        description=(
            "The grape leaf appears healthy with no visible signs of disease. "
            "Leaves show normal green coloration, intact structure, and no "
            "lesions, spots, or discoloration indicative of pathological conditions."
        ),
        symptoms=[
            "Uniform green coloration across the leaf blade",
            "No spots, lesions, or necrotic areas present",
            "Normal leaf shape and margin integrity",
            "Clear venation without chlorosis or striping",
            "No wilting, curling, or premature senescence",
        ],
        treatment=[
            "No treatment required for healthy foliage",
            "Continue regular monitoring and scouting schedule",
            "Maintain current vineyard management practices",
            "Document as baseline for comparison in future inspections",
        ],
        prevention=[
            "Maintain balanced irrigation and nutrition programs",
            "Implement integrated pest and disease management (IPDM)",
            "Keep records of vineyard health for trend analysis",
            "Sanitize tools and equipment to prevent pathogen introduction",
            "Monitor environmental conditions that favor disease development",
        ],
        severity="low",
    ),
}


def get_disease_info(class_name: str) -> DiseaseInfo:
    """Retrieve disease information by class display name."""
    if class_name in DISEASE_DATABASE:
        return DISEASE_DATABASE[class_name]

    # Fuzzy match by slug or partial name
    normalized = class_name.lower().replace("_", " ")
    for key, info in DISEASE_DATABASE.items():
        if key.lower() in normalized or normalized in key.lower():
            return info

    return DiseaseInfo(
        name=class_name,
        description=f"No detailed information available for '{class_name}'.",
        symptoms=["Information not available"],
        treatment=["Consult a plant pathologist or agricultural extension service"],
        prevention=["Regular monitoring and proper vineyard hygiene"],
        severity="medium",
    )
