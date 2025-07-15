import pandas as pd
import pytest
from pytest import param

from pyvallib.cfi.opm import OPM, CapitalStructure

err_msg_col = "DataFrame missing following required columns: {'seniority'}"
err_msg_class_name = "class_name column must be unique"
err_msg_type = "type column must contain only Preferred, Common values"
err_msg_cap_structure = "cap_structure must be an instance of CapitalStructure"
err_msg_target_class = "Class Series G Preferred not found in capital structure"

# 1. Convertible Preferred
prefA = ["Series A Preferred", "Preferred", 100000, 6, 10, True, 1, False, None]
prefB = ["Series B Preferred", "Preferred", 150000, 5, 25, True, 1, False, None]
prefC = ["Series C Preferred", "Preferred", 200000, 4, 50, True, 1, False, None]
prefD = ["Series D Preferred", "Preferred", 225000, 3, 100, True, 1, False, None]
prefE = ["Series E Preferred", "Preferred", 250000, 2, 400, True, 1, False, None]
prefF = ["Series F Preferred", "Preferred", 300000, 1, 500, True, 1, False, None]
common = ["Common", "Common", 800000, None, None, None, None, None, None]
data1 = [prefA, prefB, prefC, prefD, prefE, prefF, common]
fv1 = [324.2992, 324.5484, 325.6714, 331.1009, 449.4766, 524.9271, 324.2607]
vol1 = [0.5927, 0.5915, 0.5864, 0.5630, 0.2723, 0.1584, 0.5929]


# 2. Participating Preferred
prefA = ["Series A Preferred", "Preferred", 100000, 6, 10, True, 1, True, None]
prefB = ["Series B Preferred", "Preferred", 150000, 5, 25, True, 1, True, None]
prefC = ["Series C Preferred", "Preferred", 200000, 4, 50, True, 1, True, None]
prefD = ["Series D Preferred", "Preferred", 225000, 3, 100, True, 1, True, None]
prefE = ["Series E Preferred", "Preferred", 250000, 2, 400, True, 1, True, None]
prefF = ["Series F Preferred", "Preferred", 300000, 1, 500, True, 1, True, None]
common = ["Common", "Common", 800000, None, None, None, None, None, None]
data2 = [prefA, prefB, prefC, prefD, prefE, prefF, common]
fv2 = [250.6868, 263.2861, 284.5354, 328.0232, 599.7895, 703.0742, 242.3206]
vol2 = [0.6523, 0.6271, 0.5887, 0.5223, 0.3012, 0.2334, 0.6703]


# 3. Participating Preferred with participation cap
prefA = ["Series A Preferred", "Preferred", 100000, 1, 10, True, 1, True, 2]
prefB = ["Series B Preferred", "Preferred", 150000, 1, 25, True, 1, True, 2]
prefC = ["Series C Preferred", "Preferred", 200000, 1, 50, True, 1, True, 3]
prefD = ["Series D Preferred", "Preferred", 225000, 1, 100, True, 1, True, 3]
common = ["Common", "Common", 800000, None, None, None, None, None, None]
data3 = [prefA, prefB, prefC, prefD, common]
fv3 = [505.5461, 505.5669, 507.2661, 523.2035, 505.5454]
vol3 = [0.4586, 0.4585, 0.4519, 0.4095, 0.4586]


# 4. Convertible Preferred with 2x liquidation preference
prefA = ["Series A Preferred", "Preferred", 100000, 1, 20, True, 1, False, None]
prefB = ["Series B Preferred", "Preferred", 150000, 1, 50, True, 1, False, None]
prefC = ["Series C Preferred", "Preferred", 200000, 1, 100, True, 1, False, None]
prefD = ["Series D Preferred", "Preferred", 225000, 1, 200, True, 1, False, None]
common = ["Common", "Common", 800000, None, None, None, None, None, None]
data4 = [prefA, prefB, prefC, prefD, common]
fv4 = [195.0236, 196.8733, 204.4566, 242.1574, 194.4874]
vol4 = [0.5020, 0.4850, 0.4292, 0.2637, 0.5076]

# 5. Convertible preferred with acrruing dividends that convert
prefA = ["Series A Preferred", "Preferred", 100000, 1, 15.2664, True, 1.5266, False, None]
prefB = ["Series B Preferred", "Preferred", 150000, 1, 34.0122, True, 1.3605, False, None]
prefC = ["Series C Preferred", "Preferred", 200000, 1, 61.7853, True, 1.2357, False, None]
prefD = ["Series D Preferred", "Preferred", 225000, 1, 103.9230, True, 1.0392, False, None]
common = ["Common", "Common", 800000, None, None, None, None, None, None]
data5 = [prefA, prefB, prefC, prefD, common]
fv5 = [76.7340, 82.1918, 91.3886, 109.9887, 71.2158]
vol5 = [0.5233, 0.4646, 0.3632, 0.1978, 0.5744]

# 6. Convertible preferred with accruing dividends that don't convert
prefA = ["Series A Preferred", "Preferred", 100000, 1, 15.2664, True, 1, False, None]
prefB = ["Series B Preferred", "Preferred", 150000, 1, 34.0122, True, 1, False, None]
prefC = ["Series C Preferred", "Preferred", 200000, 1, 61.7853, True, 1, False, None]
prefD = ["Series D Preferred", "Preferred", 225000, 1, 103.9230, True, 1, False, None]
common = ["Common", "Common", 800000, None, None, None, None, None, None]
data6 = [prefA, prefB, prefC, prefD, common]
fv6 = [74.6961, 77.9798, 87.0138, 109.8703, 73.3873]
vol6 = [0.5377, 0.4764, 0.3569, 0.1971, 0.5677]

# 7. Test missing column
data7 = pd.DataFrame(data6, columns=CapitalStructure.cols).rename(columns={"seniority": "test"})

# 8. Test duplicate class_name
data8 = [prefA] + data1

# 9. Test error in type
prefA_err = prefA.copy()
prefA_err[1] = "Test"
data9 = [prefA_err, prefB, prefC, common]


@pytest.mark.parametrize(
    "inputs, error_message",
    [
        # Test with missing column
        param({"data": data7}, err_msg_col, id="test_w_missing_col"),
        # Test with duplicate class_name
        param({"data": data8, "columns": CapitalStructure.cols}, err_msg_class_name, id="test_w_duplicate_class_name"),
        # Test with invalid type
        param({"data": data9, "columns": CapitalStructure.cols}, err_msg_type, id="test_w_type"),
    ],
)
def test_capitalstructure_errs(inputs, error_message):
    with pytest.raises(ValueError) as e:
        CapitalStructure(**inputs)
    assert error_message in str(e)


cols = CapitalStructure.cols


@pytest.mark.parametrize(
    "cap_structure, equity, T, vol, r, expected, error_message",
    [
        # Test normal methods and properties
        param(CapitalStructure(data1, columns=cols), 750000000, 2, 0.45, 0.04, [fv1, vol1], None, id="test_opm_1"),
        param(CapitalStructure(data2, columns=cols), 750000000, 2, 0.45, 0.04, [fv2, vol2], None, id="test_opm_2"),
        param(CapitalStructure(data3, columns=cols), 750000000, 2, 0.45, 0.04, [fv3, vol3], None, id="test_opm_3"),
        param(CapitalStructure(data4, columns=cols), 300000000, 2, 0.45, 0.04, [fv4, vol4], None, id="test_opm_4"),
        param(CapitalStructure(data5, columns=cols), 120000000, 2, 0.45, 0.04, [fv5, vol5], None, id="test_opm_5"),
        param(CapitalStructure(data6, columns=cols), 120000000, 2, 0.45, 0.04, [fv6, vol6], None, id="test_opm_6"),
        # Test cap_structure input
        param(data6, None, None, None, None, None, err_msg_cap_structure, id="test_cap_structure"),
        # # Test array with different dimensions
        # param([10, 16], [10, 12, 15], 5, 0.45, 0.05, 0, [], err_msg_array_dim, id="test_array_S_K"),
    ],
)
def test_opm_output(cap_structure, equity, T, vol, r, expected, error_message):
    if error_message is None:
        opm = OPM(cap_structure)
        output = opm.model(equity, T, vol, r)
        assert output["fair_value_share"].to_list() == pytest.approx(expected[0], abs=1e-3)
        assert output["series_specific_vol"].to_list() == pytest.approx(expected[1], abs=1e-3)

    else:
        with pytest.raises(TypeError) as e:
            opm = OPM(cap_structure)
        assert error_message in str(e)


@pytest.mark.parametrize(
    "cap_structure, target_value, target_class, T, vol, r, expected, error_message",
    [
        # Test backsolve method
        param(
            CapitalStructure(data1, columns=cols),
            150000000,
            "Series F Preferred",
            2,
            0.45,
            0.04,
            639220356.8456,
            None,
            id="test_backsolve_1",
        ),
        param(
            CapitalStructure(data2, columns=cols),
            150000000,
            "Series F Preferred",
            2,
            0.45,
            0.04,
            331951672.3675,
            None,
            id="test_backsolve_2",
        ),
        # Test target class not in capital structure
        param(
            CapitalStructure(data1, columns=cols),
            150000000,
            "Series G Preferred",
            2,
            0.45,
            0.04,
            639220356.8456,
            err_msg_target_class,
            id="test_target_class",
        ),
    ],
)
def test_opm_backsolve(cap_structure, target_value, target_class, T, vol, r, expected, error_message):
    if error_message is None:
        opm = OPM(cap_structure)
        backsolve = opm.backsolve(target_value, target_class, T, vol, r)
        assert backsolve == pytest.approx(expected, abs=1e-2)

    else:
        with pytest.raises(ValueError) as e:
            opm = OPM(cap_structure)
            backsolve = opm.backsolve(target_value, target_class, T, vol, r)
        assert error_message in str(e)
