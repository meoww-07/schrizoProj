import numpy as np
import pandas as pd
import pytest

from src.audit_dataset import (REASONS, UnmappedValueError, add_exclusions, load_visual_qc, map_value,
                               normalize_value, parse_number, sample_flow, value_lookup)
from src.extract_functional_features import roi_network
from src.extract_structural_features import read_stats

LABELS = {"patient": ["Patient"], "control": ["Control"], "exclude": ["Disenrolled"]}


def manifest_rows():
    base = {"label": 1, "age": 30.0, "sex_male": 1, "n_t1": 1, "n_bold": 1, "fs_complete": True,
            "struct_visual_qc": 1, "n_preproc_bold": 1, "func_qc_pass": True, "strict_patient": 1}
    variants = [
        {},                                      # included patient
        {"label": 0, "strict_patient": 0},       # included control
        {"label": np.nan},                       # missing label
        {"n_t1": 0, "fs_complete": False},       # missing structural (also fails structural QC)
        {"struct_visual_qc": np.nan},            # visual QC not recorded
        {"func_qc_pass": False},                 # motion failure
    ]
    return pd.DataFrame([{**base, "participant_id": f"sub-{i}", **v} for i, v in enumerate(variants)])


def test_exclusions_and_sequential_flow():
    m = add_exclusions(manifest_rows())
    assert m["included"].tolist() == [1, 1, 0, 0, 0, 0]
    assert m.loc[3, "exclusion_reasons"] == "missing_structural;failed_structural_qc"
    flow = dict(zip(*sample_flow(m).T.values))
    assert flow["initial_N"] == 6
    assert flow["excluded_missing_label"] == 1
    assert flow["excluded_excluded_label"] == 0
    assert flow["excluded_missing_structural"] == 1
    assert flow["excluded_failed_structural_qc"] == 1   # sub-3 already counted above
    assert flow["excluded_failed_functional_qc"] == 1
    assert flow["final_N"] == m["included"].sum() == 2
    assert (flow["patients"], flow["controls"]) == (1, 1)
    assert sum(flow[f"excluded_{r}"] for r in REASONS) + flow["final_N"] == flow["initial_N"]


def test_excluded_labels_are_counted_separately_from_missing_labels():
    rows = manifest_rows()
    rows["label_status"] = ["patient", "control", "missing", "patient", "patient", "patient"]
    rows.loc[1, "label_status"] = "excluded"
    flow = dict(zip(*sample_flow(add_exclusions(rows)).T.values))
    assert flow["excluded_missing_label"] == 1
    assert flow["excluded_excluded_label"] == 1


@pytest.mark.parametrize("raw", [None, np.nan, pd.NA, "", "   ", "nan", "NaN", "None", "unknown",
                                 " Not Reported ", "n/a"])
def test_missing_values_map_to_none(raw):
    assert normalize_value(raw) is None
    assert map_value(raw, LABELS, "diagnosis") is None


@pytest.mark.parametrize("raw", ["Patient", "patient", "PATIENT", "  patient  "])
def test_matching_ignores_case_and_whitespace(raw):
    assert map_value(raw, LABELS, "diagnosis") == "patient"


def test_explicitly_excluded_label_is_not_missing():
    assert map_value("disenrolled", LABELS, "diagnosis") == "exclude"


def test_unrecognized_value_raises_instead_of_becoming_missing():
    with pytest.raises(UnmappedValueError, match="Pateint"):
        map_value("Pateint", LABELS, "diagnosis")


def test_numeric_codes_match_integer_config_values():
    sex = {"male": [0], "female": [1]}
    assert map_value(0.0, sex, "sex") == "male"
    assert map_value(np.int64(1), sex, "sex") == "female"
    assert map_value("1", sex, "sex") == "female"
    with pytest.raises(UnmappedValueError):
        map_value(2.0, sex, "sex")


def test_config_rejects_ambiguous_or_missing_token_entries():
    with pytest.raises(ValueError, match="both"):
        value_lookup({"patient": ["Patient"], "control": ["patient"]})
    with pytest.raises(ValueError, match="missing-value token"):
        value_lookup({"patient": ["unknown"], "control": ["Control"]})


def test_custom_missing_tokens():
    assert normalize_value("-9", missing_values=["-9"]) is None
    assert normalize_value(-9, missing_values=["-9"]) is None
    assert normalize_value("unknown", missing_values=["-9"]) == "unknown"


def test_parse_number():
    assert parse_number(" 34 ", "age") == 34.0
    assert parse_number("unknown", "age") is None
    assert parse_number(np.nan, "age") is None
    with pytest.raises(UnmappedValueError, match="not a number"):
        parse_number("3O", "age")


def test_visual_qc_accepts_only_pass_fail_or_empty(tmp_path):
    good = tmp_path / "qc.csv"
    good.write_text("participant_id,qc_pass,rater,notes\n0040000,1,A,\nsub-0040001,0,A,\n0040002,,,\n")
    ratings = load_visual_qc(good)
    assert ratings["sub-0040000"] == 1 and ratings["sub-0040001"] == 0 and np.isnan(ratings["sub-0040002"])
    bad = tmp_path / "bad.csv"
    bad.write_text("participant_id,qc_pass,rater,notes\n0040000,yes,A,\n")
    with pytest.raises(ValueError, match="qc_pass"):
        load_visual_qc(bad)


def test_read_freesurfer_stats(tmp_path):
    path = tmp_path / "aseg.stats"
    path.write_text(
        "# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated Total Intracranial Volume, 1545362.45, mm^3\n"
        "# Measure SurfaceHoles, SurfaceHoles, Total number of defect holes in surfaces prior to fixing, 42, unitless\n"
        "# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean\n"
        "  1   4   5000   5010.2  Left-Lateral-Ventricle  30.1\n"
        "  2  10   7000   7055.0  Left-Thalamus            90.2\n"
    )
    measures, table = read_stats(path)
    assert measures["eTIV"] == 1545362.45 and measures["SurfaceHoles"] == 42
    assert table.loc[table["StructName"] == "Left-Thalamus", "Volume_mm3"].item() == 7055.0


def test_roi_network_parses_schaefer_labels():
    assert roi_network("7Networks_LH_Vis_1") == "Vis"
    assert roi_network("7Networks_RH_SalVentAttn_Med_2") == "SalVentAttn"
    assert roi_network("Background") == "NA"
