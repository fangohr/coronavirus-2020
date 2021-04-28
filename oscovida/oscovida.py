"""Code used for notebooks and data exploration on
https://github.com/oscovida/oscovida"""


import datetime
import math
import os
import pytz
import time
import joblib
import numpy as np
import pandas as pd
import IPython.display
from typing import Tuple, Union
# choose font - can be deactivated
from matplotlib import rcParams
from oscovida.data_sources import base_url, hungary_data, jhu_population_url, rki_data, rki_population_url
from oscovida.plotting_helpers import align_twinx_ticks, cut_dates, has_twin, limit_to_smoothed, uncertain_tail, \
    full_extent

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
rcParams['svg.fonttype'] = 'none'
# need many figures for index.ipynb and germany.ipynb
rcParams['figure.max_open_warning'] = 50
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter, date2num, MONDAY, WeekdayLocator
from matplotlib.ticker import ScalarFormatter, FuncFormatter, FixedLocator
from bisect import bisect

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# suppress warning
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

LW = 3   # line width

# set up joblib memory to avoid re-fetching files
joblib_location = "./cachedir"
joblib_memory = joblib.Memory(joblib_location, verbose=0)


def compute_binder_link(notebook_name):
    """Given a string """
    root_url = "https://mybinder.org/v2/gh/oscovida/binder/master?filepath=ipynb/"
    return root_url + notebook_name


def display_binder_link(notebook_name):
    url = compute_binder_link(notebook_name)
    # print(f"url is {url}")
    IPython.display.display(
        IPython.display.Markdown(f'[Execute this notebook with Binder]({url})'))


def clear_cache():
    """Need to run this before new data for the day is created"""
    joblib_memory.clear()


def double_time_exponential(q2_div_q1, t2_minus_t1=None):
    """ See https://en.wikipedia.org/wiki/Doubling_time"""
    if t2_minus_t1 is None:
        t2_minus_t1 = np.ones(q2_div_q1.shape)
    return t2_minus_t1 * np.log(2) / np.log(q2_div_q1)


def report_download(url, df):
    print(f"Downloaded data: last data point {df.columns[-1]} from {url}")


@joblib_memory.cache
def fetch_deaths_last_execution():
    """Use to remember at what time and date the last set of deaths was downloaded.
    A bit of a hack as we didn't know how to get this out of joblib.
    """
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


@joblib_memory.cache
def fetch_cases_last_execution():
    """See fetch_deaths_last_execution"""
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


@joblib_memory.cache
def fetch_deaths():
    """Download deaths from Johns Hopkins data repository"""
    url = os.path.join(base_url, "time_series_covid19_" + "deaths" + "_global.csv")
    df = pd.read_csv(url, index_col=1)
    report_download(url, df)
    fetch_deaths_last_execution()
    return df


@joblib_memory.cache
def fetch_deaths_US():
    """Download deaths for US states from Johns Hopkins data repository"""
    url = os.path.join(base_url, "time_series_covid19_" + "deaths" + "_US.csv")
    df = pd.read_csv(url, index_col=1)
    report_download(url, df)
    # fetch_deaths_last_execution_()
    return df


@joblib_memory.cache
def fetch_cases():
    """Download cases from Johns Hopkins data repository"""
    url = os.path.join(base_url, "time_series_covid19_" + "confirmed" + "_global.csv")
    df = pd.read_csv(url, index_col=1)
    report_download(url, df)
    fetch_cases_last_execution()
    return df


@joblib_memory.cache
def fetch_cases_US():
    """Download cases for US status from Johns Hopkins data repository"""
    url = os.path.join(base_url, "time_series_covid19_" + "confirmed" + "_US.csv")
    df = pd.read_csv(url, index_col=1)
    report_download(url, df)
    fetch_cases_last_execution()
    return df


def get_country_data_johns_hopkins(country: str,
                                   region: str = None, subregion: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Given a country name, return deaths and cases as a tuple of
    pandas time series. Works for all (?) countries in the world, or at least
    those in the Johns Hopkins data set. All rows should contain a datetime
    index and a value.
    """

    deaths = fetch_deaths()
    cases = fetch_cases()

    assert country in deaths.index, f"{country} not in available countries. These are {sorted(deaths.index)}"

    # Some countries report sub areas (i.e. multiple rows per country) such as China, France, United Kingdom
    # Denmark. In that case, we sum over all regions.
    tmp = deaths.loc[country]
    if len(tmp.shape) == 1:
        d = deaths.loc[country]
    elif len(tmp.shape) == 2:   # China, France, United Kingdom, ...
        d = deaths.loc[country].sum()
        d.rename("deaths", inplace=True)
    else:
        raise ValueError("Unknown data set structure for deaths {country}:", tmp)

    tmp = cases.loc[country]
    if len(tmp.shape) == 1:
        c = cases.loc[country]
    elif len(tmp.shape) == 2:
        c = cases.loc[country].sum()
        c.rename("cases", inplace=True)
    else:
        raise ValueError("Unknown data set structure for cases {country}:", tmp)

    # make date string into timeindex
    d.index = pd.to_datetime(d.index, errors="coerce", format="%m/%d/%y")
    c.index = pd.to_datetime(c.index, errors="coerce", format="%m/%d/%y")
    # drop all rows that don't have data
    # sanity check: how many do we drop?
    if c.index.isnull().sum() > 3:
        print(f"about to drop {c.index.isnull().sum()} entries due to NaT in index", c)
    c = c[c.index.notnull()]

    if d.index.isnull().sum() > 3:
        print(f"about to drop {d.index.isnull().sum()} entries due to NaT in index", d)
    d = d[d.index.notnull()]

    # check there are no NaN is in the data
    assert c.isnull().sum() == 0, f"{c.isnull().sum()} NaNs in {c}"
    assert d.isnull().sum() == 0, f"{d.isnull().sum()} NaNs in {d}"

    # label data
    c.name = country + " cases"
    d.name = country + " deaths"

    return c, d


def get_US_region_list():
    """return list of strings with US state names"""
    deaths = fetch_deaths_US()
    return list(deaths.groupby("Province_State").sum().index)


def get_region_US(state, county=None, debug=False):
    """Given a US state name and county, return deaths and cases as a tuple of pandas time
    series. (Johns Hopkins data set)

    If country is None, then sum over all counties in that state (i.e. return
    the numbers for the state.)

    """

    if not county is None:
        raise NotImplementedError("Can only process US states (no counties)")

    deaths = fetch_deaths_US()
    cases = fetch_cases_US()

    assert state in deaths['Province_State'].values, \
        f"{state} not in available states. These are {sorted(set(deaths['Province_State']))}"

    if county is None:
        tmpd = deaths.groupby('Province_State').sum()
        d = tmpd.loc[state]
        tmpc = cases.groupby('Province_State').sum()
        c = tmpc.loc[state]
    else:
        raise NotImplementedError("Can't do counties yet.")
    # Some countries report sub areas (i.e. multiple rows per country) such as China, France, United Kingdom
    # Denmark. In that case, we sum over all regions.

    # make date string into timeindex
    d.index = pd.to_datetime(d.index, errors="coerce", format="%m/%d/%y")
    c.index = pd.to_datetime(c.index, errors="coerce", format="%m/%d/%y")
    # drop all rows that don't have data
    # sanity check: how many do we drop?
    if c.index.isnull().sum() > 3:
        if debug:
            print(f"about to drop {c.index.isnull().sum()} entries due to NaT in index", c)
    c = c[c.index.notnull()]

    if d.index.isnull().sum() > 3:
        if debug:
            print(f"about to drop {d.index.isnull().sum()} entries due to NaT in index", d)
    d = d[d.index.notnull()]

    # check there are no NaN is in the data
    assert c.isnull().sum() == 0, f"{c.isnull().sum()} NaNs in {c}"
    assert d.isnull().sum() == 0, f"{d.isnull().sum()} NaNs in {d}"

    # label data
    country = f"US-{state}"
    c.name = country + " cases"
    d.name = country + " deaths"

    return c, d


def compose_dataframe_summary(cases, deaths):
    """Used in per-country template to show data table.
    Could be extended.

    Expects series of cases and deaths (time-aligned), combines those in DataFrame and returns it
    """
    df = pd.DataFrame()
    df["total cases"] = cases
    df["daily new cases"] = cases.diff()
    if deaths is not None:
        df["total deaths"] = deaths
        df["daily new deaths"] = deaths.diff()

    # drop first row with nan -> otherwise ints are shows as float in table
    df = df.dropna().astype(int)

    # change index: latest numbers shown first
    df = df[::-1]

    return df


@joblib_memory.cache
def fetch_data_germany_last_execution():
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


@joblib_memory.cache
def fetch_data_germany(include_last_day=True) -> pd.DataFrame:
    """Fetch data for Germany from Robert Koch institute and return as a pandas
    DataFrame with the `Meldedatum` as the index.

    Data source is https://npgeo-corona-npgeo-de.hub.arcgis.com . The text on that
    webpage implies that the data comes from the Robert Koch Institute.

    As an option (`include_last_day`), we can omit the last day with data from
    the retrieved data sets (see reasoning below in source), as the data is
    commonly update one day later with more accurate (and typically higher)
    numbers.
    """

    # outdated: datasource = "https://opendata.arcgis.com/datasets/dd4580c810204019a7b8eb3e0b329dd6_0.csv"
    datasource = rki_data
    t0 = time.time()
    print(f"Please be patient - downloading data from {datasource} ...")
    germany = pd.read_csv(datasource)
    delta_t = time.time() - t0
    print(f"Completed downloading {len(germany)} rows in {delta_t:.1f} seconds.")

    ## create new column 'landkreis' and get rid of "SK " and "LK " for this
    ## - this is too simplistic. We have fields like "Region Hannover"
    # germany['landkreis'] = germany['Landkreis'].apply(lambda s: s[3:])

    # (at least) the last data from the Robert-Koch-Institute (RKI) seems not to be
    # fully reported the day after. For example, on 3 April, the number of cases
    # from RKI is well below what is expected. Example:
    #
    # From RKI (as of evening of 2020-04-03:)
    # 2020-03-29    62653
    # 2020-03-30    66692
    # 2020-03-31    72333
    # 2020-04-01    77464
    # 2020-04-02    79625
    #
    # From Johns Hopkins (as of evening of 2020-04-03:):
    # 2020-03-29    62095
    # 2020-03-30    66885
    # 2020-03-31    71808
    # 2020-04-01    77872
    # 2020-04-02    84794
    #
    # So we must assume that the RKI data will be corrected later; maybe the next day.
    #
    # To make our plots not inaccurate, we'll remove the last data point from the RKI data:
    g2 = germany.set_index(pd.to_datetime(germany['Meldedatum']))
    g2.index.name = 'date'

    # get rid of last day in data if desired
    if include_last_day == False:
        last_day = g2.index.max()
        sel = g2.index == last_day
        cleaned = g2.drop(g2[sel].index, inplace=False)
    else:
        cleaned = g2

    fetch_data_germany_last_execution()
    return cleaned


def pad_cumulative_series_to_yesterday(series):
    """Given a time series with date as index and cumulative cases/deaths as values:

    - if the last date in the index is older than yesterday, then
    - add that date
    - resample the series with a daily interval, using padding with last known value
    - and return.

    Required for Robert Koch Data, where only a new data point is provided if
    the numbers change, but the plotting algorithms need to know that there is
    no change. Without this padding, the data set looks old as the last plotted
    data point is the last one for which data is provided.
    """
    now = datetime.datetime.now()
    rki_tz = pytz.timezone('Europe/Berlin')
    now_tz = datetime.datetime.now(rki_tz)

    # remove time zone information from datetime, so we can compare against
    # datatime dates from get_country_data which has no timezone information
    # attached.
    now = now.replace(tzinfo=None)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday = today - pd.Timedelta("1D")
    last = series.index.max()
    if last < yesterday:
        # repeat last data point with index for yesterday
        series[yesterday] = series[last]
        series2 = series.resample("1D").pad()
        return series2
    else:
        return series


def germany_get_region(state=None, landkreis=None, pad2yesterday=False):
    """ Returns cases and deaths time series for Germany, and a label for the state/kreis.

    If state is given, return sum of cases (as function of time) in that state (state=Bundesland)

    If Landkreis is given, return data from just that Landkreis.

    Landkreis seems unique, so there is no need to provide state and Landkreis.

    [Should tidy up names here; maybe go to region and subregion in the function argument name, and
    translate later.]
    """
    germany = fetch_data_germany()
    """Returns two time series: (cases, deaths)"""
    assert state or landkreis, "Need to provide a value for state or landkreis"

    if state and landkreis:
        raise NotImplementedError("Try to use 'None' for the state.")
        #  TODO: We need to check if this is important.

    if state:
        if not state in germany['Bundesland'].values:
            raise Exception(
                f"{state} not in available German states. These are "
                f"{sorted(germany['Bundesland'].drop_duplicates())}"
            )

        land = germany[germany['Bundesland'] == state]
        land = land.set_index(pd.to_datetime(land['Meldedatum']))
        land.index.name = 'date'
        land.sort_index(inplace=True)

        # group over multiple rows for the same date
        # (this will also group over the different landkreise in the state)
        cases = land["AnzahlFall"].groupby('date').agg('sum').cumsum()
        region_label = f'Germany-{state}'
        cases.name = region_label + " cases"

        # group over all multiple entries per day
        deaths = land["AnzahlTodesfall"].groupby('date').agg('sum').cumsum()
        deaths.name = region_label + " deaths"

        if pad2yesterday:
            deaths = pad_cumulative_series_to_yesterday(deaths)
            cases = pad_cumulative_series_to_yesterday(cases)

        return cases, deaths

    if landkreis:
        assert landkreis in germany['Landkreis'].values, \
            f"{state} not in available German states. These are {sorted(germany['Landkreis'].drop_duplicates())}"

        lk = germany[germany["Landkreis"] == landkreis]
        lk.index = pd.to_datetime(lk['Meldedatum'])
        lk.index.name = 'date'
        lk = lk.sort_index()

        cases = lk["AnzahlFall"].groupby('date').agg('sum').cumsum()
        region_label = f'Germany-{landkreis}'
        cases.name = region_label + ' cases'

        deaths = lk["AnzahlTodesfall"].groupby('date').agg('sum').cumsum()
        deaths.name = region_label + ' deaths'

        if pad2yesterday:
            deaths = pad_cumulative_series_to_yesterday(deaths)
            cases = pad_cumulative_series_to_yesterday(cases)

        return cases, deaths


@joblib_memory.cache
def fetch_csv_data_from_url(source) -> pd.DataFrame:
    """Given a URL, fetch the csv using pandas. Put into separate function (from germany_get_population)
    to avoid repeated download of file (for better performance)."""
    data = pd.read_csv(source)
    return data # type: ignore


@joblib_memory.cache
def germany_get_population() -> pd.DataFrame:
    """The function's behavior duplicates the one for `germany_get_region()` one."""
    source = rki_population_url
    population = fetch_csv_data_from_url(source)
    population = (
        population
        .set_index('county')
    )
    population = population.rename(columns={"EWZ": "population"})
    # see https://github.com/oscovida/oscovida/issues/210
    # try to remove this if-clause and see if tests fail:
    if "LK Saar-Pfalz-Kreis" in population.index:
        population.loc['LK Saarpfalz-Kreis'] = population.loc['LK Saar-Pfalz-Kreis']
        population = population.drop('LK Saar-Pfalz-Kreis')
    return population # type: ignore


@joblib_memory.cache
def get_population() -> pd.DataFrame:
    """Returns a DataFrame with population data"""
    source = jhu_population_url
    population = fetch_csv_data_from_url(source)
    #  Only include population of entire countries by excluding rows that belong
    #  to a province or state
    population = population[population['Province_State'].isnull()]
    population = population[population["Population"] != 0]
    population = (
        population
        .groupby('Country_Region')
        .sum() # Here we sum over individual regions in a country to get the total country population
        .rename(columns={"Population": "population"}) # Rename Population to population
    )
    return population # type: ignore


def population(country: str,
               region: str = None, subregion: str = None) -> Union[int, None]:
    """
    Returns an `int` which corresponds to the population.
    Only supports JHU countries and Germany Landkreise so far.

    Example:

    $> population(country="Germany", subregion="LK Pinneberg")
    316103
    $> population(country="France")
    65273512
    """
    df = fetch_csv_data_from_url(jhu_population_url)\
        .rename(columns={"Population": "population",
                         "Country_Region": "country",
                         "Province_State": "region",
                         "Admin2": "subregion"})
    df = df[df['population'].notnull()]

    if country in df.country.values:
        if region is None and subregion is None:
            # use JHU data
            return int(df[(df['country'] == country)
                          & (df['region'].isnull())
                          & (df['subregion'].isnull())
                          ].population)
        elif region and subregion:
            raise NotImplementedError("Cannot use both region and subregion")

        elif country.casefold() == 'germany':
            if region or subregion:
                df = germany_get_population()
                if region in df['BL'].values:
                    return int(df[df['BL'] == region].population.sum())
                elif subregion in df.index:
                    return int(df.population[subregion])
                elif region in df.index:    # silently try to use as subregion
                    return int(df.population[region])
                else:
                    raise NotImplementedError(f"{region} in neither in available German Lands nor in Landkreises. " \
                                              f"These are {', '.join(sorted(df['BL'].drop_duplicates()))} for Lands and " \
                                              f"{', '.join(sorted(df.index))} for Landkreises.")
        else:
            if region or subregion:
                if region in df['region'].values:
                    combined_key = f"{region}, {country}"
                    if combined_key in df['Combined_Key'].values:  # the total population of the region is known
                        return int(df[df['Combined_Key'] == combined_key].population)
                    else:   # there's no total population in the dataset, we have to sum up on our own
                        return int(df[df['region'] == region].population.sum())
                elif subregion in df['subregion'].values:
                    return int(df.population[subregion])
                elif region in df['subregion']:    # silently try to use as subregion
                    return int(df.population[subregion])
                else:
                    return
    else:
        return


@joblib_memory.cache
def get_incidence_rates_countries(period=14):
    cases = fetch_cases()
    deaths = fetch_deaths()

    cases = cases.groupby(cases.index).sum().astype(int)
    deaths = deaths.groupby(deaths.index).sum().astype(int)

    #  Sanity checks that the column format is as expected
    assert all(cases.columns == deaths.columns)
    assert all(cases.columns[:2] == ["Lat", "Long"])

    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    fortnight_ago = yesterday - datetime.timedelta(days=period+1)
    periods = (fortnight_ago < pd.to_datetime(cases.columns[2:])) & (
        pd.to_datetime(cases.columns[2:]) < yesterday
    )
    periods = np.concatenate((np.full(2, False), periods))  # Add the 3 ignored columns

    assert len(periods) == len(cases.columns)

    cases_sum = (
        cases[cases.columns[periods]]
        .diff(axis=1)
        .sum(axis=1).astype(int)
        .to_frame(f"{period}-day-sum")
    )
    deaths_sum = (
        deaths[deaths.columns[periods]]
        .diff(axis=1)
        .sum(axis=1).astype(int)
        .to_frame(f"{period}-day-sum")
    )

    # Now we have tables like:
    #                     14-day-sum
    # Country/Region
    # Afghanistan              841.0
    # ...                        ...
    # Zimbabwe                1294.0
    #
    # [266 rows x 1 columns]
    # Where the values are the total cases/deaths in the past `period` days

    population = (
        get_population()
        .population.astype(int)
    )

    cases_incidence = cases_sum.join(population)
    cases_incidence.index.name = "Country"
    cases_incidence[f"{period}-day-incidence-rate"] = (
        cases_incidence[f"{period}-day-sum"] / cases_incidence["population"] * 100_000
    ).round(decimals=1)
    deaths_incidence = deaths_sum.join(population)
    deaths_incidence.index.name = "Country"
    deaths_incidence[f"{period}-day-incidence-rate"] = (
        deaths_incidence[f"{period}-day-sum"] / deaths_incidence["population"] * 100_000
    ).round(decimals=1)

    # We could join the tables, but it's easier to join them than to split so
    # we'll just return two instead
    return cases_incidence, deaths_incidence


@joblib_memory.cache
def get_incidence_rates_germany(period=14):
    germany = fetch_data_germany()
    germany = germany.rename(
        columns={"AnzahlFall": "cases", "AnzahlTodesfall": "deaths"}
    )

    yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
    fortnight_ago = yesterday - datetime.timedelta(days=period)
    periods = (fortnight_ago < germany.index) & (germany.index < yesterday)
    germany = germany.iloc[periods]

    index = germany[['Bundesland', 'Landkreis']]

    index['Landkreis'] = 'Germany: ' + index['Landkreis']
    index['Bundesland'] = '(' + index['Bundesland'] + ')'

    germany['metadata-index'] = index['Landkreis'] + ' ' + index['Bundesland']

    cases_sum = (
        germany.groupby("Landkreis")
        .agg({'cases': 'sum', 'Bundesland': 'first', 'metadata-index': 'first'})
        .rename(columns={"cases": f"{period}-day-sum"})
    )
    deaths_sum = (
        germany.groupby("Landkreis")
        .agg({'deaths': 'sum', 'Bundesland': 'first', 'metadata-index': 'first'})
        .rename(columns={"deaths": f"{period}-day-sum"})
    )

    # Now we have tables like:
    #                            cases
    # Landkreis
    # LK Ahrweiler                       32
    # ...                               ...
    # StadtRegion Aachen                109
    # [406 rows x 1 columns]
    # Where the values are the total cases/deaths in the past `period` days

    population = germany_get_population().population.astype(int).to_frame()

    cases_incidence = cases_sum.join(population)
    cases_incidence[f"{period}-day-incidence-rate"] = (
        cases_incidence[f"{period}-day-sum"] / cases_incidence["population"] * 100_000
    ).round(decimals=1)
    # one-line-summary is used for the index entry on the table
    cases_incidence['one-line-summary'] = (cases_incidence.index
        .map(lambda x:  x[3:] + " (LK)" if x.startswith("LK ") else x)
        .map(lambda x:  x[3:] + " (SK)" if x.startswith("SK ") else x)
        + ' (' + cases_incidence['Bundesland'] + ')'
    )

    deaths_incidence = deaths_sum.join(population)
    deaths_incidence[f"{period}-day-incidence-rate"] = (
        deaths_incidence[f"{period}-day-sum"] / deaths_incidence["population"] * 100_000
    ).round(decimals=1)
    deaths_incidence['one-line-summary'] = (deaths_incidence.index
        .map(lambda x:  x[3:] + " (LK)" if x.startswith("LK ") else x)
        .map(lambda x:  x[3:] + " (SK)" if x.startswith("SK ") else x)
        + ' (' + deaths_incidence['Bundesland'] + ')'
    )

    # We could join the tables, but it's easier to join them than to split so
    # we'll just return two instead
    return cases_incidence, deaths_incidence



@joblib_memory.cache
def fetch_data_hungary_last_execution():
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")


@joblib_memory.cache
def fetch_data_hungary():
    """
    Fetch data for Hungary from https://github.com/sanbrock/covid19

    Dataset does not contain the number of deaths in each county/capital city.
    """
    datasource = hungary_data

    t0 = time.time()
    print(f"Please be patient - downloading data from {datasource} ...")
    hungary = pd.read_csv(datasource)
    delta_t = time.time() - t0
    print(f"Completed downloading {len(hungary)} rows in {delta_t:.1f} seconds.")

    # Dropping the last row, because it is a duplicate of header.
    hungary.drop(hungary.index[-1], inplace=True)
    # Will be at least one date with no values.
    hungary.dropna(inplace=True)

    hungary = hungary.astype({col_name: int for col_name in hungary.columns[1:]})

    fetch_data_hungary_last_execution()
    return hungary


def get_counties_hungary():
    # return fetch_data_hungary().columns[1:]
    return ['Bács-Kiskun', 'Baranya', 'Békés', 'Borsod-Abaúj-Zemplén', 'Budapest', 'Csongrád', 'Fejér',
            'Győr-Moson-Sopron', 'Hajdú-Bihar', 'Heves', 'Jász-Nagykun-Szolnok', 'Komárom-Esztergom', 'Nógrád', 'Pest',
            'Somogy', 'Szabolcs-Szatmár-Bereg', 'Tolna', 'Vas', 'Veszprém', 'Zala']


def get_region_hungary(county):
    """
    Returns cases int time series and label for county in Hungary.

    """
    counties_and_the_capital_city = get_counties_hungary()

    if county not in counties_and_the_capital_city:
        raise ValueError(f'{county} must be one of: \n{counties_and_the_capital_city}')

    hungary = fetch_data_hungary()

    hungary.set_index(pd.to_datetime(hungary['Dátum']), inplace=True)
    cases = hungary[county]
    region_label = f'Hungary-{county}'
    cases.name = region_label + " cases"

    return cases, None


def plot_time_step(ax, series, style="-", labels=None, logscale=True):
    """Plot the series as cumulative cases/deaths, plotted as step function.
    Parameters:
    - ax : axis object from matplotlib
    - series: the actual data as pandas Series
    - style : matplotlib style/Colour
    - labels: tuple of (region_name, kind) where
              both elements are strings, and 'kind' is either 'cases' or 'deaths'.
    - logscale: plot y-axis in logscale (default=True)

    Returns ax object.
    """

    ax.step(series.index, series.values, style, label=series.name,
           linewidth=LW)
    if logscale:
        ax.set_yscale('log')
    ax.legend()
    ax.set_ylabel("total numbers")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    return ax


def plot_incidence_rate(ax, cases: pd.Series, country: str,
                        region: str = None, subregion: str = None,
                        dates: str = None, weeks: int = 0):
    if dates:
        cases = cut_dates(cases, dates)
    if weeks:
        cases = cases[-7 * (weeks+1):]  # we need to add one week, because we loose one on rolling

    habitants = population(country=country, region=region, subregion=subregion)

    if habitants:
        incidence = (cases.diff().dropna().rolling(7).sum()/habitants*100000)
        norm_title = "\n(per 100K people)"
    else:
        incidence = (cases.diff().dropna().rolling(7).sum())
        norm_title = ''

    # convert dates to numbers first
    inxval = date2num(incidence.index.to_pydatetime())
    # smoothing
    if len(incidence.values) > 100:
        # the number of consecutive segments of the same color, bigger number produces less spaces between segments:
        chain_len = 2   # 2 means that each segment of the curve is of length 2
    else:
        chain_len = 1
    # a little trick here: we can't split odd amount of points into even number of segments of length 2, therefore
    # we simply throw away the very first point of the curve making the number of points even
    starting_point = len(inxval) % chain_len
    points = np.array([inxval[starting_point:], incidence.values[starting_point:]]).T.reshape(-1, chain_len, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # colormap: the line is green on the bottom, near zero value, then it's yellow around 30,
    # then in becomes more red, and above 100 it's totally maroon
    colors = ["green", "gold", "red", "maroon"]
    cmap = LinearSegmentedColormap.from_list('RedGreen', colors, len(incidence))
    norm = plt.Normalize(0, 100)
    if country == "Germany" and not dates:  # see https://github.com/oscovida/oscovida/issues/228
        # plot the end of the incidence curve dotted for Germany (the last four points)
        lc = LineCollection(segments[:-int(4/chain_len)], cmap=cmap, norm=norm, linewidth=LW)
        lc_end = LineCollection(segments[-int(4/chain_len):], cmap=cmap, norm=norm, linewidth=3, linestyles=[':'], alpha=0.5)
        lc_end.set_array(incidence.values[-int(4/chain_len)::chain_len])  # set the colors according to y values
        ax.add_collection(lc_end)
    else:
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=LW)
    lc.set_array(incidence.values[::chain_len])  # set the colors according to y values

    # add collection to axes
    ax.add_collection(lc)
    ax.autoscale_view()
    ax.set_ylabel("7-day incidence rate"+norm_title)
    ax.yaxis.set_major_formatter(ScalarFormatter())

    x, y = incidence.index[-1], incidence.values[-1]
    ax.plot([x], [y], marker="+", color=cmap(norm(y)))  # marker at the end of the line
    ax.annotate(f"{y:.1f}", xy=(x, y * 1.1), color=cmap(norm(y)))   # marker annotation
    return ax


def compute_daily_change(series: pd.Series) -> Tuple[Tuple[pd.Series, str],
                                                     Tuple[pd.Series, str],
                                                     Tuple[pd.Series, str]]:
    """returns (change, smooth, smooth2)

    where 'change' is a tuple of (series, label)
    and smooth is a tuple of (series, label).
    and smooth2 is a tuple of (series, label).

    'change' returns the raw data (with nan's dropped)
    'smooth' makes the data smoother
    'smooth2' does some additional smoothing - more artistic than scientific

    The 'change' under consideration, is the day-to-day change of the series.
    We assume that there is one entry per day in the Series.

    """
    diff = series.diff().dropna()
    label = ""
    change = diff, label

    # smoothed curve, technical description
    smooth_label = f"Gaussian window (stddev=3 days)"
    # shorter description
    smooth_label = f"(rolling 7d mean)"

    # Smoothing step 1: get rid of weekly cycles
    rolling_series = diff.rolling(7, center=True,
                                  win_type=None,
                                  min_periods=7).mean()
    smooth = rolling_series, smooth_label

    # extra smoothing for better visual effects
    rolling_series2 = rolling_series.rolling(7, center=True,
                                             win_type='gaussian',
                                             min_periods=7).mean(std=3)

    # extra smooth curve
    smooth2_label = "Smoothed " + smooth_label
    # shorter description
    smooth2_label = smooth_label

    smooth2 = rolling_series2, smooth2_label

    return change, smooth, smooth2


def plot_daily_change(ax, series: pd.Series, color: str, labels: Tuple[str, str] = None,
                      country: str = None, region: str = None, subregion: str = None,
                      dates: str = None, weeks: int = 0, uncertainty: int = 7):
    """Given a series of data and matplotlib axis ax, plot the
    - difference in the series data from day to day as bars and plot a smooth
    - line to show the overall development

    - series is pandas.Series with data as index, and cumulative cases (or
    deaths)
    - `color` is color to be used for plotting
    - `dates` or `weeks` specify the time range to plot
    - `uncertainty` is a number of days we plot as a dashed line

    See plot_time_step for documentation on other parameters.
    """
    bar_alpha = 0.2
    if labels is None:
        type_label = ""
        region_label = ""
    else:
        region_label, type_label = labels
    ax_label = region_label + " new " + type_label
    (change, change_label), _, (smooth2, smooth2_label) = compute_daily_change(series)
    if dates:
        change = cut_dates(change, dates)
        smooth2 = cut_dates(smooth2, dates)
        uncertainty = 1
    if weeks:
        change = change[-7 * weeks:]
        smooth2 = smooth2[-7 * weeks:]

    if country and (region or subregion):
        habitants = population(country=country, region=region, subregion=subregion)
    else:
        habitants = population(region_label)
    if habitants and max(change) > 0:
        # create another Y-axis on the right hand side
        # unfortunately there's no simple way of swapping the axes,
        # therefore we define normalised axis first
        ax2 = ax
        # this is just to be sure that we plot the same graph (please leave it commented in the production):
        # ax2.plot(smooth2.index, smooth2.values * 1E5 / habitants, color='green')
        ax2.set_ylim((0, max(change) * 1E5 / habitants))
        ax2.set_ylabel('daily change\nnormalised per 100K')

        ax1 = ax2.twinx()  # create an independent y-axis
        ax1.set_ylim((0, max(change)))

        ticks = align_twinx_ticks(ax_left=ax2, ax_right=ax1)
        ax1.yaxis.set_major_locator(FixedLocator(ticks))
        ax1.bar(change.index, change.values, color=color,
                label=ax_label, alpha=bar_alpha, linewidth=LW)
        # RKI data for deaths in German regions is very untrustworthy,
        # see https://oscovida.github.io/2020-germany-reporting-delay-meldeverzug.html
        if country == "Germany" and (region or subregion) and labels[1] == "deaths" and not dates:
            uncertainty = 6 * 7     # six weeks

        ax1.plot(smooth2.index[:-uncertainty], smooth2.values[:-uncertainty], color=color,
                 label=ax_label + " " + smooth2_label, linewidth=LW)
        # uncertain (dashed) line on the right
        if uncertainty > 1:
            uncertain_tail(ax1, series[-2 * uncertainty:].diff().dropna(), days=uncertainty,
                           color=color, linewidth=LW, alpha=0.7)

        ax1.legend()
        ax1.set_ylabel('daily change')

    else:
        ax.tick_params(left=True, right=True, labelleft=True, labelright=True)
        ax.yaxis.set_ticks_position('both')
        ax.plot(smooth2.index, smooth2.values, color=color,
                label=ax_label + " " + smooth2_label, linewidth=LW)
        ax.bar(change.index, change.values, color=color,
                label=ax_label, alpha=bar_alpha, linewidth=LW)

        ax.legend()
        ax.set_ylabel('daily change')


    # Data cleaning: France new cases daily reports had a huge spike
    # on 12 April 2020 with 26849 reported new cases. This would set
    # too large a ymax for a while, which was fixed manually by setting
    # ymax to min(10000, ymax)... until in October 2020 daily reported
    # cases soared beyond that anomalous reported value of 12 April 2020.
    # France also has a negative value of circa -2000 on 22 April 2020.
    # We limit the y-scale to correct manually for this:
    if region_label == "France" and type_label == "cases":
        # get current limits
        ymin, ymax = ax.get_ylim()
        ax.set_ylim([max(-500, ymin), ymax])

    return ax


def compute_doubling_time(series, minchange=0.5, labels=None, debug=False):

    """
    Compute and return doubling time of (assumed exponential) growth, based on two
    data points. We use data points from subsequent days.

    returns (dtime, smooth)

    Where 'dtime' is a tuple of (series, label)
    and smooth is a tuple of (series, label).

    'dtime' returns the raw data (with nan's dropped)
    'smooth' makes the data smoother

    The 'dtime' under consideration, is the day-to-day dtime of the series.
    We assume that there is one entry per day in the Series.

    If there is not enough data to compute the doubling time, returns
    ((None, message), (None, None)) where 'message' provides
    data for debugging the analysis.
    """

    if labels is None:
        label = ""
        region = ""
    else:
        region, label = labels


    # only keep values where there is a change of a minumum number
    # get rid of data points where change is small values
    (f, f_label) , (change_smoothed, smoothed_label), _ = compute_daily_change(series)
    sel = change_smoothed < minchange
    reduced = series.drop(f[sel].index, inplace=False)
    if len(reduced) <= 1:   # no data left
        return (None, "no data in reduced data set"), (None, None)

    ratio = reduced.pct_change() + 1  # computes q2/q1 =
    ratio_smooth = reduced.rolling(7, center=True, win_type='gaussian',
                                   min_periods=7).mean(std=3).pct_change() + 1

    if debug:
        print(f"len(ratio) = {len(ratio.dropna())}, {ratio}")
        print(f"len(ratio_smooth) = {len(ratio_smooth.dropna())}, {ratio_smooth}")


    # can have np.inf and np.nan at this point in ratio
    # if those are the only values, then we should stop
    ratio.replace(np.inf, np.nan, inplace=True)
    if ratio.isna().all():
        return (None, "Cannot compute ratio"), (None, None)

    ratio_smooth.replace(np.inf, np.nan, inplace=True)
    if ratio_smooth.isna().all():
        # no useful data in smooth line, but data for dots is okay
        # Give up anyway
        return (None, "Cannot compute smooth ratio"), (None, None)

    # computes q2/q1
    # compute the actual doubling time
    dtime = double_time_exponential(ratio, t2_minus_t1=1)
    dtime_smooth = double_time_exponential(ratio_smooth, t2_minus_t1=1)

    if debug:
        print(f"len(dtime) = {len(dtime.dropna())}, {dtime}")
        print(f"len(dtime_smooth) = {len(dtime_smooth.dropna())}, {dtime_smooth}")

    # can have np.inf and np.nan at this point in dtime_smooth and dtime
    # if those are the only values, then we should stop
    dtime_smooth.replace(np.inf, np.nan, inplace=True)
    if dtime_smooth.isna().all():
        # We could at this point carry on and return the dtime, but not dtime_smooth.
        # This may not be a common use case and not worth the extra complications.
        return (None, "Cannot compute doubling time"), (None, None)

    dtime.replace(np.inf, np.nan, inplace=True)
    if dtime.isna().all():
        return (None, "Cannot compute smooth doubling time"), (None, None)

    dtime_label = region + " doubling time " + label
    dtime_smooth_label = dtime_label + ' 7-day rolling mean (stddev=3)'
    # simplified label
    dtime_smooth_label = dtime_label + ' (rolling mean)'

    return (dtime, dtime_label), (dtime_smooth, dtime_smooth_label)


@limit_to_smoothed
def plot_doubling_time(ax, series, color, minchange=0.5, labels=None, debug=False):
    """Plot doubling time of series, assuming series is accumulated cases/deaths as
    function of days.

    Returns axis.

    See plot_time_step for documentation on other parameters.
    """

    if labels is None:
        labels = "", ""
    region, label = labels

    (dtime, dtime_label), (dtime_smooth, dtime_smooth_label) = \
        compute_doubling_time(series, minchange=minchange, debug=debug, labels=labels)

    if dtime is None:
        if debug:
            print(dtime_label)
        return ax

    ax.plot(dtime.index, dtime.values, 'o', color=color, alpha=0.3, label=dtime_label)

    # good to take maximum value from here
    dtime_smooth.replace(np.inf, np.nan, inplace=True)  # get rid of x/0 results, which affect max()
    # ymax = min(dtime_smooth.max()*1.5, 5000)  # China has doubling time of 3000 in between

    ## Adding a little bit of additional smoothing just for visual effects
    dtime_smooth2 = dtime_smooth.rolling(3, win_type='gaussian', min_periods=1, center=True).mean(std=1)

    # ax.set_ylim(0, max(ymax, ax.get_ylim()[1]))     # since we combine two plots, let's take another one into account
    ax.plot(dtime_smooth2.index, dtime_smooth2.values, "-", color=color, alpha=1.0,
            label=dtime_smooth_label,
            linewidth=LW)
    ax.set_ylabel(f"{labels[1]}\ndoubling time [days]")
    return ax


def compute_growth_factor(series):
    """returns (growth, smooth)

    where 'growth' is a tuple of (series, label)
    and smooth is a tuple of (series, label).

    'growth' returns the raw data (with nan's dropped)
    'smooth' makes the data smoother

    """

    # start from smooth diffs as used in plot 1
    (change, change_label) , (smooth, smooth_label), \
        (smooth2, smooth2_label) = compute_daily_change(series)

    # Compute ratio of yesterday to day
    f = smooth.pct_change() + 1  # compute ratio of subsequent daily changes
                                 # f for growth Factor
    label = ""
    growth = (f, label)

    # division by zero may lead to np.inf in the data: get rid of that
    f.replace(np.inf, np.nan, inplace=True)  # seems not to affect plot

    # pct_change computed between two values that are NaN each, returns 0
    # (which should be interpreted as: no change from yesterday's NaN to
    #  today's NaN. However, the way we use it, this gives misleading values. )
    # Let's take out those values from f, where the input data was NaN:
    f[smooth.isna()]=np.nan

    # Compute smoother version for line in plots
    f_smoothed = f.rolling(7, center=True, win_type='gaussian', min_periods=3).mean(std=2)
    smooth_label = f"Gaussian window (stddev=2 days)"
    # simplified label
    smooth_label = "(rolling mean)"

    smoothed = f_smoothed, smooth_label

    return growth, smoothed


#def plot_growth_factor(ax, series, color):
#    """relative change of number of new cases/deaths from day to day
#    See https://youtu.be/Kas0tIxDvrg?t=330, 5:30 onwards
#
#    Computed based smooth daily change data.
#    """
#
#    # get smooth data from plot 1 to base this plot on
#    (f, f_label) , (f_smoothed, smoothed_label) = compute_growth_factor(series)
#
#
#    label = series.country + " " + series.label + " growth factor " + f_label
#    ax.plot(f.index, f.values, 'o', color=color, alpha=0.3, label=label)
#
#    label = series.country + " " + series.label + " growth factor " + smoothed_label
#    ax.plot(f_smoothed.index, f_smoothed.values, '-', color=color, label=label, linewidth=LW)
#
#    # ax.legend(loc='lower left')
#    ax.legend()
#    ax.set_ylabel("growth factor")
#    ax.set_ylim(0.5, 1.5)  # should generally be below 1
#    ax.plot([series.index.min(), series.index.max()], [1.0, 1.0], '-C3') # label="critical value"
#    return ax


# Computation or R
#
def compute_R(daily_change, tau=4):
    """Given a time series s, estimate R based on description from RKI [1].

    [1] [Robert Koch Institute: Epidemiologisches Bulletin 17 | 2020 23. April 2020]
    https://www.rki.de/DE/Content/Infekt/EpidBull/Archiv/2020/Ausgaben/17_20.html

    Steps:

    1. Compute change from day to day
    2. Take tau-day averages (tau=4 is recommended as of April/May 2020)
    3. divide average from days 4 to 7 by averages from day 0 to 3, and use this data point for day[7]

    """
    # change = s.diff()
    change = daily_change
    mean4d = change.rolling(tau).mean()
    R = mean4d / mean4d.shift(tau)
    R2 = R.shift(-tau)  # this is not the RKI method, but seems more appropriate:
                        # we centre the reported value between the 2-intervals of length tau
                        # that have been used to compute it.

    # Can we create an R-value of 1.0 for small numbers (approaching 0)
    # of new cases/deaths? At least if we have no new cases, then
    # R=1 seems a reasonable outcome.
    R2[(mean4d.shift(tau) == 0.0) & (mean4d == 0)] = 1.0

    return R2


def min_max_in_past_n_days(series, n, at_least = [0.75, 1.25], alert=[0.1, 100], print_alert=False):
    """Given a time series, find the min and max values in the time series within the last n days.

    If those values are within the interval `at_least`, then use the values in at_least as the limits.
    if those values are outside the interval `at_least` then exchange the interval accordingly.

    If the values exceed the min and max value in 'alerts', then print an error message.
    Return min, max.
    """
    if n > len(series):
        n = len(series)

    series = series.replace(math.inf, math.nan)
    series = series.replace(-math.inf, math.nan)

    min_ = series[-n:].min() - 0.1    # the -0.1 is to make extra space because the line we draw is thick
    max_ = series[-n:].max() + 0.1

    if min_ < at_least[0]:
        min_final = min_
    else:
        min_final = at_least[0]

    if max_ > at_least[1]:
        max_final = max_
    else:
        max_final = at_least[1]

    if print_alert:
        if max_final > alert[1]:
            # print(f"Large value for R_max = {max_final} > {alert[1]} in last {n} days: \n", series[-n:])
            print(f"Large value for R_max = {max_final} > {alert[1]} in last {n} days: \n")
        if min_final < alert[0]:
            # print(f"Small value for R_min = {min_final} < {alert[0]} in last {n} days: \n", series[-n:])
            print(f"Small value for R_min = {min_final} < {alert[0]} in last {n} days: \n")


    # print(f"DDD: min_={min_}, max_={max_}")
    return min_final, max_final


def plot_reproduction_number(ax, series, color_g='C1', color_R='C4',
                             yscale_days=28, max_yscale=10, smoothing=0,
                             labels=None):
    """
    - series is expected to be time series of cases or deaths
    - label is 'cases' or 'deaths' or whatever is desired as the description
    - country is the name of the region/country
    - color_g is the colour for the growth factor
    - color_R is the colour for the reproduction number

    R number is calculated based on a 7-day average of the daily changes.
    See details in compute_R().

    By default the resulting data is not smoothed further. If desired,
    for visual purposes, the parameter `smoothing` can be given the number
    of days over which the computer R-values are averaged out (using a Gaussian
    window with a width of 3 days). A number of 5 or 7 gives smooth results.
    However, the plotted data then stops 5/2 (=2 days) or 7/2 (=3) days earlier.

    See plot_time_step for documentation on other parameters.
    """

    if labels is None:
        region, label = "", ""
    else:
        region, label = labels

    # get smooth data for growth factor from plot 1 to base this plot on
    (f, f_label), (f_smoothed, smoothed_label) = compute_growth_factor(series)

    label_ = region + " " + label + " daily growth factor " + f_label
    ax.plot(f.index, f.values, 'o', color=color_g, alpha=0.3, label=label_)

    label_ = region + " " + label + " daily growth factor " + smoothed_label
    ax.plot(f_smoothed.index, f_smoothed.values, '-', color=color_g,
            label=label_, linewidth=LW, alpha=0.7)


    # # data for computation or R
    # start from smooth diffs as used in plot 1
    (change, change_label) , (smooth, smooth_label), \
        (smooth2, smooth2_label) = compute_daily_change(series)

    # we only use one return value (the 7-day average)
    smooth_diff = smooth
    R = compute_R(smooth_diff)
    # do some additional smoothing
    if smoothing != 0:
        R = R.rolling(smoothing, win_type='gaussian', center=True,
                      min_periods=smoothing).mean(std=3)
    ax.plot(R.index, R, "-", color=color_R,
            label=region + f" estimated R (using {label})",
            linewidth=4.5, alpha=1)

    # choose y limits so that all data points of R in the last 28 days are visible
    min_, max_ = min_max_in_past_n_days(R, yscale_days);

    # set upper bound for R
    # (Germany data has huge spike in February )
    if max_ > max_yscale:
        max_ = max_yscale

    ax.set_ylim([min_, max_]);

    # Plot ylim interval for debugging
    # ax.plot([R.index.min(), R.index.max()], [min_, min_], 'b-')
    # ax.plot([R.index.min(), R.index.max()], [max_, max_], 'b-')


    ax.set_ylabel(f"R & growth factor\n(based on {label})")
    # plot line at 0
    ax.plot([series.index.min(), series.index.max()], [1.0, 1.0], '-C3') # label="critical value"
    ax.legend()
    return ax


def get_region_label(country: str, region: str = None, subregion: str = None) -> str:
    """
    Produce unique and unambiguous name from region and subregion
    """
    # TODO: this might be used to decouple the label creation and fetching data
    _country = country.casefold()
    region_label = None
    if _country == 'germany':
        # if region == None and subregion == None:
        #     region_label = _country
        if region is not None and subregion is None:
            region_label = f"Germany-{region}"
        elif region is None and subregion is not None:
            region_label = f"Germany-{subregion}"
        elif region is None and subregion is not None:
            region_label = f"Germany-{subregion}"

    elif _country == 'us' and region != None:
        region_label = f"United States: {region}"

    elif _country == 'hungary':
        # region -> térség
        # subregion -> kistérség
        # county -> megye

        if region is not None:
            region_label = f"Hungary-{region}"

    # finally:
    if not region_label:
        region_label = country

    return region_label


def get_country_data(country: str, region: str = None, subregion: str = None, verbose: bool = False,
                     pad_RKI_data_to_yesterday: bool = True) -> Tuple[pd.Series, pd.Series]:
    """Given the name of a country, get the Johns Hopkins data for cases and deaths,
    and return them as a tuple of pandas.Series objects and a string describing the region:
    (cases, deaths, region_label)

    If the country is "Germany", use the region (=Land) and subregion (=Kreis)
    to select the appropriate subset from the Robert Koch Institute. If only
    the region is provided, the data from all subregions in that region is
    accumulated.

    If the country is "US", get US data (states are available as regions) from Johns Hopkins
    repository.

    Returns "cases, deaths, country_region" where country region is a string
    describing the country and region.

    The series are resampled at a daily interval. Missing data points are replaced
    by the last provided value (seems reasonable for a data set representing
    cumulative numbers as a function of time: where no new data point is provided,
    assume the change was zero, thus the last data point can be re-used).

    Data from Johns Hopkins is reported daily, even if there is no change in numbers.

    Data from Robert Koch Institute (RKI) is only provided if the cumulative
    numbers change. We thus resample the data set if the last provided data
    point is not from yesterday, up to yesterday (which is the most recent day
    for which data could be available).

    Note that some data sets are updated retrospectively (in particular data
    from RKI), so the numbers for the a particular date may increase after one
    or two days (or even later in extreme cases).

    """
    special_countries = {
        'germany': (germany_get_region, {'state': region, 'landkreis': subregion,
                                         'pad2yesterday': pad_RKI_data_to_yesterday}),
        'us': (get_region_US, {'state': region}),
        'hungary': (get_region_hungary, {'county': region})
    }

    if country.casefold() in special_countries and (region is not None or subregion is not None):
        func, kwargs = special_countries[country.casefold()]
        c, d = func(**kwargs)
    else:
        c, d = get_country_data_johns_hopkins(country)  # use johns hopkins data

    len_cases1 = len(c)
    # hungarian county data doesn't contain number of deaths
    len_deaths1 = 0 if d is None else len(d)
    # resample data so we have one value per day
    c = c.resample("D").pad()
    d = None if d is None else d.resample("D").pad()

    len_cases2 = len(c)
    len_deaths2 = 0 if d is None else len(d)

    if verbose:
        print(f"get_country_data: cases [{len_cases1}] -> [{len_cases2}]")
        print(f"get_country_data: deaths[{len_deaths1}] -> [{len_deaths2}]")
    return c, d


def day0atleast(v0: int, series: pd.Series) -> pd.Series:
    try:
        day0 = series[series > v0].index[0]
    except IndexError:  # means no days found for which series.values > v0
        # print(f"Haven't found value > {v0} in {series.name}")
        result = pd.Series(dtype=object)    # return an empty object
        return result
    # compute timedelta
    timedelta = series.index - day0
    # convert to int as index
    t = pd.to_numeric(timedelta.astype("timedelta64[D]").astype(int))
    # Assemble new series
    result = pd.Series(index=t, data=series.values)
    return result


def align_sets_at(v0: int, df: pd.DataFrame) -> pd.DataFrame:
    """Accepts data frame, and aligns so that all entries close to v0 are on the same row.

    Returns new dataframe with integer index (representing days after v0).
    """
    res = pd.DataFrame()
    for col in df.columns:
        # res[col] = day0for(v0, df[col])
        series = day0atleast(v0, df[col])
        series.name = col
        res = pd.merge(res, series, how='outer', left_index=True, right_index=True)

    return res


def get_compare_data(countrynames, rolling=7):
    """Given a list of country names, return two dataframes: one with cases and one with deaths
    where
    - each column is one country
    - data in the column is the diff of accumulated numbers
    - any zero values are removed for italy (data error)
    - apply some smoothing
    """
    df_c = pd.DataFrame()
    df_d = pd.DataFrame()

    for countryname in countrynames:
        c, d = get_country_data_johns_hopkins(countryname)

        df_c[countryname] = c.diff().rolling(rolling, center=True).mean()  # cases
        df_d[countryname] = d.diff().rolling(rolling, center=True).mean()  # deaths

    return df_c, df_d


def plot_logdiff_time(ax, df, xaxislabel=None, yaxislabel=None, style="", labels=True, labeloffset=2, v0=0,
                      highlight={}, other_lines_alpha=0.4):
    """highlight is dictionary: {country_name : color}

    - df: DataFrame with time series to align
    """

    for i, col in enumerate(df.columns):
        if col in highlight:
            alpha = 1.0
            color = highlight[col]
            linewidth = 4
        else:
            alpha = other_lines_alpha
            # have only 10 colours
            color = style + 'C' + str(i % 10)
            linewidth = 2

        ax.plot(df.index, df[col].values, color, label=col, linewidth=linewidth, alpha=alpha)
        tmp = df[col].dropna()
        if len(tmp) > 0:  # possible we have no data points
            x, y = tmp.index[-1], tmp.values[-1]
            ax.plot([x], [y], "o" + color, alpha=alpha)     # dots at the end of each line
            if labels:
                # If we use the 'annotate' function on a data point with value 0 or a negative value,
                # we run into a bizarre bug that the created figure has very large dimensions in the
                # vertical direction when rendered to svg. The next line prevents this.
                #
                # Our fix/hack means that the label of the country will not be visible. That's not so bad
                # at the moment as the situation of zero new deaths causes the problem, and the infections
                # are higher and non-zero, thus we can see the country label in the infections plot.
                #
                # If this stops, we could consider choosing a data point from
                # earlier in the series to put the label there.
                #
                # The comparison for "y < 0" should not be necessary as the new deaths per day can at
                # most be zero. However, for Australia, there is a -1 reported for first of June,
                # which can lead to negative values when computing a rolling mean.
                y = np.NaN if y <= 0 else y

                # Add country/region name as text next to last data point of the line:
                ax.annotate(col, xy=(x + labeloffset, y), textcoords='data')
            else:
                ax.legend()
    ax.set_ylabel(yaxislabel)
    ax.set_xlabel(xaxislabel)
    ax.set_yscale('log')
    # use integer numbers for values > 1, and decimal presentation below
    # from https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting/33213196
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    # ax.set_xscale('log')    # also interesting
    if isinstance(df.index[0], (int, np.integer)):
        ax.set_ylim(bottom=set_y_axis_limit(df, v0))
        ax.set_xlim(left=-1)  #ax.set_xlim(-1, df.index.max())
    else:
        ax.legend(loc='upper left')
    ax.tick_params(left=True, right=True, labelleft=True, labelright=True)
    ax.yaxis.set_ticks_position('both')


def set_y_axis_limit(data, current_lim):
    """The function analyses the data set given and lowers
    the y-limit if there are data points below `current_lim`

    :param data: data to plot
    :param current_lim: initial y-axis lower limit
    :return: new y-axis lower limit
    """
    data_0 = data[data.index >= 0]  # from "day 0" only
    limits = [0.01, 0.1, 1, 10, 100]
    # if we have values within the `limits`, we set the lower `y_limit` on the graph to the value on the left of bisect
    # example: if the minimum value is 3, then y_limit = 1
    index = bisect(limits, data_0.min().min())
    if 0 < index < len(limits):
        return limits[index - 1]
    elif index == 0:
        return limits[0]
    else:
        return current_lim


def make_compare_plot(main_country, compare_with=["Germany", "Australia", "Poland", "Korea, South",
                                                  "Belarus", "Switzerland", "US"],
                     v0c=10, v0d=3, normalise=True, align=False):
    rolling = 7
    df_c, df_d = get_compare_data([main_country] + compare_with, rolling=rolling)

    if align:
        res_c = align_sets_at(v0c, df_c)
        res_d = align_sets_at(v0d, df_d)
        c_xlabel = f'days since {v0c} cases'
        d_xlabel = f'days since {v0d} deaths'
    else:
        res_c, res_d = df_c, df_d
        c_xlabel, d_xlabel = None, None

    if normalise:
        for country in res_c.keys():
            res_c[country] *= 100000 / population(country)
            res_d[country] *= 100000 / population(country)

    # We get NaNs for some lines. This seems to originate in the original data set not having a value recorded
    # for all days.
    # For the purpose of this plot, we'll just interpolate between the last and next known values.
    # We limit the number of fills to 3 days. (Just a guess to avoid accidental
    # filling of too many NaNs.)

    res_c = res_c.interpolate(method='linear', limit=3)
    res_d = res_d.interpolate(method='linear', limit=3)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax = axes[0]
    norm_str = '\nper 100K people'
    plot_logdiff_time(ax, res_c, xaxislabel=c_xlabel,
                      yaxislabel=f"daily new cases{norm_str if normalise else ''}\n(rolling 7-day mean)",
                      v0=v0c, highlight={main_country: "C1"}, labels=False)
    ax = axes[1]
    plot_logdiff_time(ax, res_d, xaxislabel=d_xlabel,
                      yaxislabel=f"daily new deaths{norm_str if normalise else ''}\n(rolling 7-day mean)",
                      v0=v0d, highlight={main_country: "C0"}, labels=False)

    if not normalise:
        fig.tight_layout(pad=1)
    title = f"Daily cases (top) and deaths (below) for {main_country}"
    axes[0].set_title(title)

    return axes, res_c, res_d


###################### Compare plots for Germany

def label_from_region_subregion(region_subregion):
    region, subregion = unpack_region_subregion(region_subregion)
    if subregion:
        if region:
            label = f"{region}-{subregion}"
        else:
            label = f"{subregion}"
    else:
        label = f"{region}"
    return label


def unpack_region_subregion(region_subregion):
    """Convention for regions in Germany (could also be useful for other countries later):

    - region_subregion is either
      - a tuple of strings (region, subregion) or
      - a string "region"

    Return a a tuple (region, subregion), where subregion is None if not provided.
    """
    if isinstance(region_subregion, tuple):
        if len(region_subregion) == 1:
            region = region_subregion[0]
            subregion = None
        elif len(region_subregion) == 2:
            region, subregion = region_subregion
        else:
            raise ValueError("region_subregion must be single value or 2-valued tuple", region_subregion)
    else:
        # assume it is just the region
        assert isinstance(region_subregion, str)
        region, subregion = region_subregion, None
    return region, subregion


def get_compare_data_germany(region_subregion, compare_with_local, rolling=7):
    """Given a region_subregion for Germany, and a list of region_subregion to compare with,
    return two dataframes: one with cases and one with deaths
    where
    - each column is one country
    - data in the column is the diff of accumulated numbers
    - any zero values are removed for italy (data error)
    - apply some smoothing

    See unpack_region_subregion for details on region_subregion.
    """
    df_c = pd.DataFrame()
    df_d = pd.DataFrame()

    for reg_subreg in [region_subregion] + compare_with_local:

        region, subregion = unpack_region_subregion(reg_subreg)
        c, d = germany_get_region(state=region, landkreis=subregion)

        label = label_from_region_subregion((region, subregion))
        df_c[label] = c.diff().rolling(rolling, center=True).mean()  # cases
        df_d[label] = d.diff().rolling(rolling, center=True).mean()  # deaths

    return df_c, df_d


def get_compare_data_hungary(region, compare_with_local: list, rolling=7):
    """Given a region for Hungary, and a list of regions to compare with,
    return two dataframes: one with cases and one with deaths
    where
    - each column is one country
    - data in the column is the diff of accumulated numbers
    - any zero values are removed for italy (data error)
    - apply some smoothing

    """
    df_c = pd.DataFrame()
    # df_d = pd.DataFrame()
    for reg in [region] + compare_with_local:
        c, d = get_region_hungary(county=reg)
        label = str(reg)
        df_c[label] = c.diff().rolling(rolling, center=True).mean()  # cases
        # df_d[label] = d.diff().rolling(rolling, center=True).mean()  # deaths
    return df_c, None


def make_compare_plot_germany(region=None, subregion=None,
                              compare_with=[],  # "China", "Italy", "Germany"],
                              compare_with_local=['Bayern',
                                                  'Berlin', 'Bremen',
                                                  'Hamburg', 'Hessen',
                                                  'Nordrhein-Westfalen',
                                                  'Sachsen-Anhalt'],
                              v0c=10, v0d=1, normalise=True,
                              weeks=0, dates=None):
    rolling = 7
    df_c1, df_d1 = get_compare_data_germany((region, subregion), compare_with_local, rolling=rolling)
    df_c2, df_d2 = get_compare_data(compare_with, rolling=rolling)

    # need to get index into same timezone before merging
    df_d1.set_index(df_d1.index.tz_localize(None), inplace=True)
    df_c1.set_index(df_c1.index.tz_localize(None), inplace=True)

    df_c = pd.merge(df_c1, df_c2, how='outer', left_index=True, right_index=True)
    df_d = pd.merge(df_d1, df_d2, how='outer', left_index=True, right_index=True)

    kwargs_c, kwargs_d = {}, {}

    if dates and weeks == 0:
        res_c = cut_dates(df_c, dates)
        res_d = cut_dates(df_d, dates)
        kwargs_c.update({'labels': False})
        kwargs_d.update({'labels': False})
    elif dates and weeks:
        raise ValueError("`dates` and `weeks` cannot be used together")
    else:
        res_c = df_c[- weeks * 7:]
        res_d = df_d[- weeks * 7:]
        kwargs_d.update({'yaxislabel': '', 'labels': False})
        kwargs_c.update({'labels': False})
        kwargs_d.update({'labels': False})

    kwargs_c.update({"yaxislabel": "daily new cases\n(rolling 7-day mean)"})
    kwargs_d.update({"yaxislabel": "daily new deaths\n(rolling 7-day mean)"})

    # We get NaNs for some lines. This seems to originate in the original data set not having a value recorded
    # for all days.
    # For the purpose of this plot, we'll just interpolate between the last and next known values.
    # We limit the number of fills to 7 days. (Just a guess to avoid accidental
    # filling of too many NaNs.)
    res_c = res_c.interpolate(method='linear', limit=7)
    res_d = res_d.interpolate(method='linear', limit=7)

    if normalise:
        kwargs_c["yaxislabel"] += "\nnormalised by 100K people"
        kwargs_d["yaxislabel"] += "\nnormalised by 100K people"
        kwargs_c.update({'labels': False})
        kwargs_d.update({'labels': False})
        for area in res_c.keys():
            res_c[area] *= 100000 / population("Germany", area)
            res_d[area] *= 100000 / population("Germany", area)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax = axes[0]
    plot_logdiff_time(ax, res_c, v0=v0c, highlight={res_c.columns[0]: "C1"}, **kwargs_c)
    ax = axes[1]

    plot_logdiff_time(ax, res_d, v0=v0d, highlight={res_d.columns[0]: "C0"}, **kwargs_d)

    # fig.tight_layout(pad=1)

    title = f"Daily cases (top) and deaths (below) for Germany: {label_from_region_subregion((region, subregion))}"
    axes[0].set_title(title)

    return axes, res_c, res_d

#######################


def choose_random_counties(exclude_region, size) -> list:
    counties = get_counties_hungary()
    assert exclude_region in counties

    counties.remove('Budapest')
    if exclude_region != 'Budapest':
        counties.remove(exclude_region)

    indices = np.arange(len(counties))
    np.random.shuffle(indices)

    counties = np.array(counties)
    choosen = counties[indices[:size]]
    choosen = list(np.concatenate((choosen, ['Budapest'])))
    return choosen


def make_compare_plot_hungary(region: str, compare_with_local: list, v0c=10):
    rolling = 7

    df_c1, _ = get_compare_data_hungary(region, compare_with_local, rolling=rolling)
    # df_c2, _ = get_compare_data(['Germany', 'Italy'], rolling=rolling)

    # need to get index into same timezone before merging
    df_c1.set_index(df_c1.index.tz_localize(None), inplace=True)
    # df_c = pd.merge(df_c1, df_c2, how='outer', left_index=True, right_index=True)

    res_c = align_sets_at(v0c, df_c1)
    res_c = res_c.interpolate(method='linear', limit=7)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_logdiff_time(axes[0], res_c, f"days since {v0c} cases",
                      "daily new cases\n(rolling 7-day mean)",
                      v0=v0c, highlight={res_c.columns[0]: "C1"}, labeloffset=0.5)

    # plot_no_data_available(axes[1], mimic_subplot=axes[0], text="daily new deaths\n(rolling 7-day mean)")
    axes[1].set_visible(False)

    title = f"Daily cases for Hungary: {label_from_region_subregion(region)}"
    axes[0].set_title(title)
    fig.tight_layout(pad=1)
    return axes, res_c, None


def plot_no_data_available(ax, mimic_subplot, text):
    # Hungary county deaths data missing
    xticks = mimic_subplot.get_xticks()
    yticks = mimic_subplot.get_yticks()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.text(xticks.mean(), yticks.mean(),
            f'No data available\n to plot {text}',
            horizontalalignment='center',
            verticalalignment='center')
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def overview(country: str, region: str = None, subregion: str = None,
             savefig: bool = False, dates: str = None,
             weeks: int = 0, data: Tuple[pd.Series, pd.Series] = None) -> Tuple[plt.axes, pd.Series, pd.Series]:
    """The `overview` function provides 6 graphs for the region:

    0) the total cumulative number of cases and deaths
    1) the daily change (cases)
    2) the daily change (deaths)
    3) R number and growth factor (cases)
    4) R number and growth factor (deaths)
    5) the doubling time (both cases and death)

    `weeks` parameter refers to the last N number of weeks to show. Omit this
    parameter or use zero value to see the pandemic since the very beginning.

    Returns: subplots, cases (pandas Series), deaths (pandas Series)
    """
    if data is None:
        _c, _d = get_country_data(country, region=region, subregion=subregion)
    else:
        _c, _d = data
        assert isinstance(_c.index[0], pd.Timestamp), f"The index of 'cases' is not of type `Timestamp`, " \
                                                      f"try to use `index=pd.DatetimeIndex(dates)`"
        assert isinstance(_d.index[0], pd.Timestamp), f"The index of 'deaths' is not of type `Timestamp`, " \
                                                      f"try to use `index=pd.DatetimeIndex(dates)`"

    region_label = get_region_label(country, region=region, subregion=subregion)
    fig, axes = plt.subplots(6, 1, figsize=(10, 15), sharex=False, constrained_layout=True)
    if dates and weeks == 0:
        c = cut_dates(_c, dates)
    elif dates and weeks:
        raise ValueError("`dates` and `weeks` cannot be used together")
    else:
        c = _c[- weeks * 7:]
    # plot_time_step(ax=axes[0], series=c, style="-C1", labels=(region_label, "cases"))
    plot_incidence_rate(ax=axes[0], cases=_c,
                        country=country, region=region, subregion=subregion, dates=dates, weeks=weeks)
    plot_daily_change(ax=axes[1], series=_c, color="C1", labels=(region_label, "cases"),
                      country=country, region=region, subregion=subregion, dates=dates, weeks=weeks)
    # data cleaning
    if country == "Spain":   # https://github.com/oscovida/oscovida/issues/44
        axes[1].set_ylim(bottom=0)
    plot_reproduction_number(axes[3], series=c, color_g="C1", color_R="C5", labels=(region_label, "cases"))
    ax_dt_c = axes[5]
    plot_doubling_time(ax_dt_c, series=c, color="C1", labels=(region_label, "cases"))
    if _d is not None:
        if dates and weeks == 0:
            d = cut_dates(_d, dates)
        else:
            d = _d[- weeks * 7:]
        # plot_time_step(ax=axes[0], series=d, style="-C0", labels=(region_label, "deaths"))
        plot_daily_change(ax=axes[2], series=_d, color="C0", labels=(region_label, "deaths"),
                          country=country, region=region, subregion=subregion, dates=dates, weeks=weeks)
        plot_reproduction_number(axes[4], series=d, color_g="C0", color_R="C4", labels=(region_label, "deaths"))
        problems = ("no data in reduced data set", "Cannot compute smooth ratio")
        if compute_doubling_time(d)[0][1] not in problems:
            ax_dt_d = plt.twinx(ax_dt_c)
            plot_doubling_time(ax_dt_d, series=d, color="C0", labels=(region_label, "deaths"))
            # combining doubling time plots
            ticks = align_twinx_ticks(ax_dt_c, ax_dt_d)
            ax_dt_d.yaxis.set_major_locator(FixedLocator(ticks))
            ax_dt_d.grid(False)     # don't draw the second grid on top of the legend
            # create a combined legend
            h_c, l_c = ax_dt_c.get_legend_handles_labels()  # may return [], []
            h_d, l_d = ax_dt_d.get_legend_handles_labels()  # may return [], []
            labels = [[], []]
            for x, y in [[h_c, l_c], [h_d, l_d]]:
                if x:
                    labels[0] += [x[1]]
                    labels[1] += [y[1]]
            axes[5].legend(*labels)
        else:   # just create a legend as is
            axes[5].legend()
    if _d is None:
        d = _d
        plot_no_data_available(axes[2], mimic_subplot=axes[1], text='daily change in deaths')
        plot_no_data_available(axes[4], mimic_subplot=axes[3], text='R & growth factor (based on deaths)')

    # enforce same x-axis on all plots
    for i in range(0, axes.shape[0]):
        axes[i].set_xlim(axes[0].get_xlim())
    for i in range(0, axes.shape[0]):
        if not has_twin(axes[i]):
            axes[i].tick_params(left=True, right=True, labelleft=True, labelright=True)
            axes[i].yaxis.set_ticks_position('both')
        if weeks > 0:
            axes[i].get_xaxis().set_major_locator(WeekdayLocator(byweekday=MONDAY))     # put ticks every Monday
            axes[i].get_xaxis().set_major_formatter(DateFormatter('%d %b'))             # date format: `15 Jun`
        else:
            axes[i].xaxis.set_major_formatter(DateFormatter("%b %y"))

    week_str = f", last {weeks} weeks" if weeks else ''
    region_str = f"{region or subregion}, {country}" if (region or subregion) else country
    title = f"{region_str}{week_str}, last data point from {c.index[-1].date().isoformat()}"
    axes[0].set_title(title)

    # tight_layout gives warnings, for example for Heinsberg
    # fig.tight_layout(pad=1)

    if savefig:
        for j in range(len(axes)):
            # get the size of the portion to save in inches
            extent = full_extent(axes[j], fig)
            filename = os.path.join("figures", region_label.replace(" ", "-").replace(",", "-") + f'_1-{j+1}.svg')
            fig.savefig(filename, bbox_inches=extent)

    return axes, c, d


def compare_plot(country: str, region: str = None, subregion: str = None,
                 savefig: bool = False, normalise: bool = True,
                 dates: str = None, align: bool = False) -> Tuple[plt.axes, pd.Series, pd.Series]:
    """ Create a pair of plots which show comparison of the region with other most suffering countries

    :param country: mandatory
    :param region: default: None
    :param subregion: default: None
    :param savefig: save figure as PDF, default: False
    :param normalise: whether to normalise data, default: True
    :param dates: range in the format "2020-03-21:2020-05-21"
    :param align: whether to align the starting point for different countries, default: False
    """
    c, d = get_country_data(country, region=region, subregion=subregion)
    region_label = get_region_label(country, region=region, subregion=subregion)
    if normalise:
        _population = population(country=country, region=region, subregion=subregion)
        c *= 100000 / _population
        d *= 100000 / _population

    if not subregion and not region:    # i.e. not a region of Germany
        axes_compare, res_c, res_d = make_compare_plot(country, normalise=normalise, align=align)

    elif country == "Germany":   # Germany specific plots
        # On 11 April, Mecklenburg Vorpommern data was missing from data set.
        # We thus compare only against those Laender, that are in the data set:
        # germany = fetch_data_germany()
        # laender = list(germany['Bundesland'].drop_duplicates().sort_values())
        axes_compare, res_c, red_d = make_compare_plot_germany(region=region, subregion=subregion, normalise=normalise,
                                                               dates=dates)
    elif country == "US" and region is not None:
        # skip comparison plot for the US states at the moment
        return None, c, d

    elif country == 'Hungary':
        # choosing random counties. not sure if this make sense or not because not every county has enough data.
        with_local = choose_random_counties(exclude_region=region, size=18)
        axes_compare, res_c, red_d = make_compare_plot_hungary(region, compare_with_local=with_local)
        return axes_compare, c, d
    else:
        raise NotImplementedError

    fig2 = plt.gcf()

    if savefig:
        for j in range(len(axes_compare)):
            # get the size of the portion to save in inches
            extent = full_extent(axes_compare[j], fig2)
            filename = os.path.join("figures", region_label.replace(" ", "-").replace(",", "-") + f'_2-{j + 1}.svg')
            fig2.savefig(filename, bbox_inches=extent)

    return axes_compare, c, d


def get_cases_last_week(cases):
    """Given cumulative cases time series, return the number of cases from the last week.
    """
    # make sure we have one value for every day
    c2 = cases.resample('D').pad()
    # last week is difference between last value, and the one 7 days before
    cases_last_week = c2[-1] - c2[-8]
    return cases_last_week
