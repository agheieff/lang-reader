from pathlib import Path

from server.language_display import language_marker, profile_option
from server.workspaces import Workspace


def test_region_markers_use_explicit_regions() -> None:
    assert language_marker("de-DE").symbol == "🇩🇪"
    assert language_marker("de-DE").description == "Germany"
    assert language_marker("es-ES").symbol == "🇪🇸"
    assert language_marker("es-ES").description == "Spain"
    assert language_marker("es-419").symbol == "🌎"
    assert language_marker("es-419").description == "Latin America and the Caribbean"


def test_chinese_markers_use_china_for_simplified_and_keep_traditional_neutral() -> None:
    simplified = language_marker("zh-Hans")
    traditional = language_marker("zh-Hant")

    assert simplified.symbol == "🇨🇳"
    assert simplified.description == "Simplified Chinese (China used as the picker marker)"
    assert traditional.symbol == "繁"
    assert traditional.description == "Traditional Han script"
    assert "Taiwan" not in traditional.description


def test_explicit_region_and_script_are_both_shown() -> None:
    marker = language_marker("zh-Hant-TW")

    assert marker.symbol == "🇹🇼 繁"
    assert marker.description == "Taiwan, Traditional Han script"
    assert language_marker("fr").symbol == "🌐"


def test_profile_option_keeps_chinese_variant_text_accessible(tmp_path: Path) -> None:
    workspace = Workspace(
        profile_id="zh-hans",
        label="Chinese (Simplified)",
        learning_language="zh-Hans",
        translation_language="en",
        directory=tmp_path,
        database_path=tmp_path / "lang.db",
        relative_directory="profiles/zh-hans",
        relative_database="profiles/zh-hans/lang.db",
    )

    option = profile_option(workspace)

    assert option.visible_label == "Chinese (Simplified)"
    assert option.accessible_label.endswith("Simplified Chinese (China used as the picker marker)")


def test_german_profile_option_uses_simple_language_name_and_flag(tmp_path: Path) -> None:
    workspace = Workspace(
        profile_id="de-de",
        label="German (Germany)",
        learning_language="de-DE",
        translation_language="en",
        directory=tmp_path,
        database_path=tmp_path / "lang.db",
        relative_directory="profiles/de-de",
        relative_database="profiles/de-de/lang.db",
    )

    option = profile_option(workspace)

    assert option.marker.symbol == "🇩🇪"
    assert option.visible_label == "German"
    assert option.accessible_label == "German; Germany"


def test_raw_registry_labels_fall_back_to_human_spanish_variants(tmp_path: Path) -> None:
    spain = Workspace(
        profile_id="es-es",
        label="es-ES",
        learning_language="es-ES",
        translation_language="en",
        directory=tmp_path / "spain",
        database_path=tmp_path / "spain" / "lang.db",
        relative_directory="profiles/es-es",
        relative_database="profiles/es-es/lang.db",
    )
    latin_america = Workspace(
        profile_id="es-419",
        label="es-419",
        learning_language="es-419",
        translation_language="en",
        directory=tmp_path / "latin-america",
        database_path=tmp_path / "latin-america" / "lang.db",
        relative_directory="profiles/es-419",
        relative_database="profiles/es-419/lang.db",
    )

    spain_option = profile_option(spain)
    latin_america_option = profile_option(latin_america)

    assert spain_option.visible_label == "Spanish (Spain)"
    assert spain_option.marker.symbol == "🇪🇸"
    assert spain_option.accessible_label == "Spanish (Spain); Spain"
    assert latin_america_option.visible_label == "Spanish (Latin America)"
    assert latin_america_option.marker.symbol == "🌎"
    assert latin_america_option.accessible_label == (
        "Spanish (Latin America); Latin America and the Caribbean"
    )


def test_known_raw_chinese_tags_never_leak_into_visible_labels(tmp_path: Path) -> None:
    options = [
        profile_option(
            Workspace(
                profile_id=f"zh-{script.casefold()}",
                label=f"zh-{script}",
                learning_language=f"zh-{script}",
                translation_language="en",
                directory=tmp_path / script,
                database_path=tmp_path / script / "lang.db",
                relative_directory=f"profiles/zh-{script.casefold()}",
                relative_database=f"profiles/zh-{script.casefold()}/lang.db",
            )
        )
        for script in ("Hans", "Hant")
    ]

    assert [option.visible_label for option in options] == [
        "Chinese (Simplified)",
        "Chinese (Traditional)",
    ]
    assert all("zh-" not in option.visible_label for option in options)
