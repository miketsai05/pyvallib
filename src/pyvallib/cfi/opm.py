import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from .blackscholes import BlackScholes


class CapitalStructure(pd.DataFrame):
    """
    Wrapper class around pd.DataFrame to represent a capital structure and ensure it has the required columns.

    Parameters:
        pd (_type_): pd.DataFrame class instance or object with the following columns:
            - class_name: unique name for each class of shares
            - type: type of share, either 'Preferred' or 'Common'
            - sharecount: number of shares for each class
            - seniority: seniority level of Preferred shares only (1 for most senior, higher numbers for less senior)
            - liquidation_preference_per_share: liquidation preference per share for Preferred shares only
            - convertible: whether the share class is convertible to common shares (True/False)
            - conversion_ratio: conversion ratio for convertible shares (None for non-convertible or Common)
            - participating: whether the share class is participating (True/False)
            - participation_cap_multiple: participation cap multiple if applicable (otherwise leave as None)
    """

    cols = {
        "class_name": "str",
        "type": "str",
        "sharecount": "int",
        "seniority": "int",
        "liquidation_preference_per_share": "float",
        "convertible": "bool",
        "conversion_ratio": "float",
        "participating": "bool",
        "participation_cap_multiple": "float",
    }

    allowed_types = ["Preferred", "Common"]

    def __init__(self, data, cols=cols, allowed_types=allowed_types, **kwargs):
        super().__init__(data, **kwargs)
        missing_cols = set(cols.keys()) - set(self.columns)
        # ensure all required columns are present
        if missing_cols:
            raise ValueError(f"DataFrame missing following required columns: {missing_cols}")
        # ensure class_name column is unique
        if not self.class_name.is_unique:
            raise ValueError("class_name column must be unique")
        # ensure type column contains only valid types
        if not self.type.isin(allowed_types).all():
            raise ValueError(f"type column must contain only {self.types} values")

        self.replace(to_replace=[None], value=np.nan, inplace=True)
        self.loc[self.convertible.isna(), "convertible"] = False
        self.loc[self.participating.isna(), "participating"] = False

    def to_df(self):
        return pd.DataFrame(self.values, columns=self.columns, index=self.index)


class OPM:
    """
    Option Pricing Model (OPM) for valuing classes of equity within capital structure

    Parameters:
        cap_structure: CapitalStructure instance containing the capital structure data
    """

    def __init__(self, cap_structure: CapitalStructure):
        if not isinstance(cap_structure, CapitalStructure):
            raise TypeError("cap_structure must be an instance of CapitalStructure")

        self.cap_structure = self._complete_cap_structure(cap_structure)

    def _complete_cap_structure(self, df: CapitalStructure):
        """
        Calculate additional columns within the capital structure to facilitate OPM calculations.
        """
        df["liquidation_preference_total"] = df.liquidation_preference_per_share * df.sharecount

        df["conversion_point"] = np.nan
        df["cap_point"] = np.nan

        # Determine conversion/participation cap points for each class in the following order:
        # Participating & No Cap - always participating - no conversion point or cap point
        # Participating & Cap - cap point
        # Participating & Cap & Convertible - conversion point
        # Non Participating & Convertible - conversion point

        # Participating & Cap - cap point
        ind = df.participating & ~df.participation_cap_multiple.isna()
        df.loc[ind, "cap_point"] = (df[ind].participation_cap_multiple - 1) * df[ind].liquidation_preference_per_share

        # Participating & Cap & Convertible - conversion point
        ind = df.participating & ~df.participation_cap_multiple.isna() & df.convertible
        df.loc[ind, "conversion_point"] = (
            df[ind].participation_cap_multiple * df[ind].liquidation_preference_per_share / df[ind].conversion_ratio
        )

        # Non Participating & Convertible
        ind = ~df.participating & df.convertible
        df.loc[ind, "conversion_point"] = df[ind].liquidation_preference_per_share / df[ind].conversion_ratio

        return df

    @property
    def _liqpref(self):
        """
        Calculate all liquidation preference breakpoints by seniority.
        """
        liqpref = pd.DataFrame(
            0.0,
            index=self.cap_structure.seniority.dropna().sort_values().unique(),
            columns=self.cap_structure.class_name.to_list(),
        )

        for i, row in liqpref.iterrows():
            ind = self.cap_structure.seniority == i
            liqpref.loc[i, self.cap_structure[ind].class_name] = self.cap_structure[
                ind
            ].liquidation_preference_total.to_list()
            liqpref.loc[i, "Description"] = (
                ", ".join(self.cap_structure[ind].class_name.to_list()) + " receive liquidation preference"
            )

        liqpref["incremental_distribution"] = liqpref[self.cap_structure.class_name].sum(axis=1)
        liqpref["type"] = "liquidation_preference"

        return liqpref

    @property
    def _interim(self):
        """
        Calculate all interim conversion or participation cap breakpoints.
        """

        interim = pd.melt(
            self.cap_structure[["class_name", "cap_point", "conversion_point"]], id_vars=["class_name"]
        ).dropna()
        interim.rename(columns={"variable": "type", "value": "common_value"}, inplace=True)
        interim.sort_values("common_value", inplace=True)

        interim.loc[:, self.cap_structure.class_name.to_list()] = 0.0
        interim.reset_index(inplace=True, drop=True)

        # Populate common sharecounts across all breakpoints
        ind = self.cap_structure.type == "Common"
        interim[self.cap_structure[ind].class_name] = self.cap_structure[ind].sharecount.to_list()

        # TO DO #
        # collapse to only unqiue and update class_name

        for i, row in interim.iterrows():
            # Factoring in participating preferred
            ind = (self.cap_structure.participating) & (self.cap_structure.cap_point.fillna(np.Inf) >= row.common_value)
            if ind.any():
                interim.loc[i, self.cap_structure[ind].class_name] = self.cap_structure[ind].sharecount.to_list()

            # Factoring in converted preferred
            ind = self.cap_structure.conversion_point < row.common_value
            if ind.any():
                interim.loc[i, self.cap_structure[ind].class_name] = self.cap_structure[ind].sharecount.to_list()

            # Populate description for row
            ind = interim.loc[i, self.cap_structure.class_name.to_list()] > 0

            if sum(ind) == len(self.cap_structure):
                des_classes = "All shares"
            else:
                des_classes = ", ".join(ind[ind].index)

            if row.type == "cap_point":
                interim.loc[
                    i, "Description"
                ] = f"{des_classes} receive distributions until {row.class_name} reaches participation cap"
            if row.type == "conversion_point":
                interim.loc[i, "Description"] = f"{des_classes} receive distributions until {row.class_name} converts"

        interim["incremental_common_value"] = np.diff(interim.common_value, prepend=0)
        interim["incremental_distribution"] = (
            interim[self.cap_structure.class_name.to_list()].sum(axis=1) * interim.incremental_common_value
        )

        return interim

    @property
    def _final(self):
        """
        Calculate final breakpoint where all shares recieve distributions
        """

        final = pd.DataFrame(0.0, index=[0], columns=self.cap_structure.class_name.to_list())

        ind = (
            (self.cap_structure.participating & self.cap_structure.participation_cap_multiple.isna())
            | self.cap_structure.convertible
            | (self.cap_structure.type == "Common")
        )
        final[self.cap_structure[ind].class_name] = self.cap_structure[ind].sharecount

        final["type"] = "all"
        final["Description"] = "All shares receive distributions pro rata"

        return final

    @property
    def breakpoint_table(self):
        """
        Calculate all breakpoints for the OPM.

        Returns:
            pd.DataFrame: DataFrame containing all breakpoints with columns:
        """

        breakpoint_table = pd.concat([self._liqpref, self._interim, self._final])
        breakpoint_table.reset_index(inplace=True, drop=True)

        breakpoint_table["From"] = [0] + breakpoint_table.incremental_distribution.cumsum().dropna().to_list()
        breakpoint_table["To"] = breakpoint_table["From"][1:].to_list() + ["and above"]

        return breakpoint_table

    @property
    def breakpoints(self):
        return self.breakpoint_table[["From", "To", "Description"]].copy()

    @property
    def allocation(self):
        """
        Calculate the distribution allocation within each breakpoint.

        Returns:
            pd.DataFrame: DataFrame containing the distribution allocation within each breakpoint.
        """

        allocation = self.breakpoint_table[self.cap_structure.class_name]
        allocation = allocation.div(allocation.sum(axis=1), axis=0)  # Normalize to get allocation percentages

        return allocation

    def model(self, equity_value: float, T: float, volatility: float, rfr: float):
        """
        Calculate the option values for each set of breakpoints based on Black-Scholes formula as a call option with
        exercise price equal to the "From" breakpoint less a call option with exercise price equal to the "To"
        breakpoint.

        Parameters:
            equity_value (float): The total equity value of the company.
            T (float): Expected time to exit or liquidity event.
            volatility (float): Volatility of the underlying equity value.
            rfr (float): Risk-free rate.

        Returns:
            pd.DataFrame: DataFrame containing the option values for each set of breakpoints.
        """

        with np.errstate(divide="ignore", invalid="ignore"):
            call_options = BlackScholes(equity_value, self.breakpoints.From, T, volatility, rfr).call_price()
        option_values = -np.diff(call_options, append=0)
        value_allocation = np.expand_dims(option_values, axis=1) * self.allocation
        fair_value_class = value_allocation.sum(axis=0)
        fair_value_share = fair_value_class / self.cap_structure.sharecount.to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            nd1 = norm.cdf(BlackScholes(equity_value, self.breakpoints.From, T, volatility, rfr).d1)
            incremental_nd1 = -np.diff(nd1, append=0)
            class_nd1 = (np.expand_dims(incremental_nd1, axis=1) * self.allocation).sum(axis=0)
            series_specific_vol = volatility * class_nd1 * equity_value / fair_value_class

        return {
            "option_values": option_values,
            "value_allocation": value_allocation,
            "fair_value_class": fair_value_class,
            "fair_value_share": fair_value_share,
            "series_specific_vol": series_specific_vol,
        }

    def backsolve(self, target_value: float, target_class: str, T: float, volatility: float, rfr: float):
        """
        Backsolve for the equity value given a target value for a specific class of shares.

        Parameters:
            target_value (float): The target value for the specified class of shares.
            target_class (str): The class of shares to backsolve for.
            T (float): Expected time to exit or liquidity event.
            volatility (float): Volatility of the underlying equity value.
            rfr (float): Risk-free rate.

        Returns:
            float: The backsolved equity value.
        """
        if target_class not in self.cap_structure.class_name.values:
            raise ValueError(f"Class {target_class} not found in capital structure.")

        def solver_func(implied_equity):
            return abs(self.model(implied_equity, T, volatility, rfr)["fair_value_class"][target_class] - target_value)

        return minimize_scalar(solver_func).x
