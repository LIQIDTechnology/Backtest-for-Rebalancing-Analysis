import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

"""
Script calculates the simulations for the liqid global portfolios.
Input for the equity returns are the regional indices of MSCI for North America, Europe, Japan, Pacific ex Japan and Emerging Markets.
Input for the bond returns are the Barclays Global Aggregate Euro-hedged Index and the Barclays Global High Yield Index.
For commodities we use Xetra Gold and the DBLCI – OY BALANCED Total Return Index.

The equities in the portfolio are GDP weighted. A fixed rebalancing takes place at the last trading day of June each year.
Another rebalancing takes place when an asset class, or sub asset class deviates to much from its target weight.

For the simulation we approximate the ETF costs by using historical TER of different ETFs.
Additionally we take liqid fees (0.15%) and trading costs (0.20%) into account.
Product cost assumptions are made based on current and historic TERs.
"""


def deduct_costs(returns, cost_dict):
    ### For case wohl nur für Spezialfall bei HQ relevant.
    # Nicht wichtig bei konsistenten Kosten. Else-Case relevant. Hier Produktkosten. Unten Tradingkosten.
    for asset, vals in cost_dict.items():
        cf = (1 - vals) ** (1 / 252)
        returns[asset] = (1 + returns[asset]) * cf - 1

        ''' Only for HQ since there were different cost at different times for single assets
        if isinstance(vals, tuple):
            (switch_date, old_cost, new_cost) = vals
            cf_old = (1 - old_cost) ** (1 / 252)
            cf_new = (1 - new_cost) ** (1 / 252)
            returns[asset] = np.where(returns[asset].index < switch_date, (1 + returns[asset]) * cf_old - 1,
                                      (1 + returns[asset]) * cf_new - 1)
        else:
            cf = (1 - vals) ** (1 / 252)
            returns[asset] = (1 + returns[asset]) * cf - 1'''

    return returns

def load_time_series_bloomberg():
    # Gibt den "schönen" bearbeiteten DataFrame aus
    # Muss wohl für jede Risikoklasse wieder Excel öffnen. Sehr langsam. Read csv wäre schneller
    #     df = pd.read_excel(r'Quelldateien\Performancedaten_Bloomberg - BBG_Formulars.xlsx', sheet_name='Liqid Global',
    df = pd.read_excel(r'Performancedaten_Bloomberg.xlsx', sheet_name='Liqid Global',
                       skiprows=[0, 1, 2, 3, 5, 6, 7], index_col=0, parse_dates=True)
    df = df.loc[df.index.dropna()]
    df = df[df.index <= '2022-03-01']  # Zeile für Testgründe um Fehler zu vermeiden

    # conversion to EUR. Global Aggregate not converted, since it is already in EUR, all others in USD
    indices_to_convert = df.columns.difference(['Bonds EUR Govt', 'High Yield World', 'Inflation Linked Bonds World','Investment Grade Corporate Bonds World', 'Equities Europe','Precious Metals', 'Euro Spot'])
    df[indices_to_convert] = df[indices_to_convert].div(df['Euro Spot'], axis=0)

    names_mapping = {'Equities North America': 'North America',
                     'Equities Europe': 'Europe',
                     'Equities Asia Pacific ex Japan': 'Pacific ex Japan',
                     'Equities Japan': 'Japan',
                     'Equities Emerging Markets World': 'Emerging Markets',
                     'Bonds EUR Govt': 'Bonds EUR Govt',
                     'Emerging Market Bonds World': 'Emerging Market Bonds World',
                     'High Yield World': 'High Yield World',
                     'Inflation Linked Bonds World': 'Inflation Linked Bonds World',
                     'Investment Grade Corporate Bonds World': 'Investment Grade Corporate Bonds World',
                     'Precious Metals': 'Gold',
                     'Commodities': 'Commodities_ex_Gold',
                     'Private Equity': 'Private Equity',
                     'Real Estate Participations': 'Real Estate Participations',
                     }

    df = df.rename(columns=names_mapping)
    df = df[names_mapping.values()]
    df = df.pct_change().dropna()

    return df


def prepare_gdp_weights(path):
    # Nimmt sich Pfad zu xslx-Datei mit weights, liest diese ein und normiert die weights auf 1 (relative).
    # Erst Unterteilung in emerging markets und developed markets, dann weiter Aufgliederung der developed markets
    # Ausgabe von den endgültigen weights
    ### Read_Excel sehr langsam und wird oft adressiert. Hier Zeitersparnis implementieren? 0.3 Sekunden für einen read. Verbesserung spart kaum Zeit.
    # Für die LGT braucht man diese Funktion nicht, da keine zeitabhängigen weights eingelesen werden müssen.
    # Change the risk class by chaning Weights_Global10 in sheet_same to Weights_GlobalXX and additionally manually change the rebalancing thresholds in the dict.
    absolute_weights = pd.read_excel(path, sheet_name='Weights_Global70', index_col=0, parse_dates=True).dropna()
    relative_weights = pd.DataFrame()

    for market in ['North America', 'Europe', 'Pacific ex Japan', 'Japan', 'Emerging Markets', 'Bonds EUR Govt', 'Emerging Market Bonds World', 'High Yield World', 'Inflation Linked Bonds World', 'Investment Grade Corporate Bonds World', 'Gold', 'Commodities_ex_Gold', 'Private Equity', 'Real Estate Participations']:
        relative_weights[market] = absolute_weights[market] / absolute_weights[['North America', 'Europe', 'Pacific ex Japan', 'Japan', 'Emerging Markets', 'Bonds EUR Govt', 'Emerging Market Bonds World', 'High Yield World', 'Inflation Linked Bonds World', 'Investment Grade Corporate Bonds World', 'Gold', 'Commodities_ex_Gold', 'Private Equity', 'Real Estate Participations']].sum(axis=1)

    # We assume that we get the GDP data for each year with a delay of 6 month.
    ### GDP Data was only relevant for HQ Strategy, since LGT is not GDP weighted and has constant weights
    # relative_weights.index = [x.replace(day=1) + pd.Timedelta(days=-1) for x in (relative_weights.index + pd.Timedelta(weeks=27))]
    return relative_weights


class Portfolio(object):
    def __init__(self, asset_returns, gdp_weights, rebalancing_thresholds, trading_costs, fees):
        self.asset_returns = asset_returns

        equities = gdp_weights['North America'].values[0] + gdp_weights['Europe'].values[0] + gdp_weights['Pacific ex Japan'].values[0] + gdp_weights['Japan'].values[0] + gdp_weights['Emerging Markets'].values[0]
        bonds = gdp_weights['Bonds EUR Govt'].values[0] + gdp_weights['Emerging Market Bonds World'].values[0] + gdp_weights['High Yield World'].values[0] + gdp_weights['Inflation Linked Bonds World'].values[0] + gdp_weights['Investment Grade Corporate Bonds World'].values[0]
        gold = gdp_weights['Gold'].values[0]
        commod = gdp_weights['Commodities_ex_Gold'].values[0]
        PE = gdp_weights['Private Equity'].values[0]
        rep = gdp_weights['Real Estate Participations'].values[0]
        self.risk_class = (equities, bonds, gold, commod, PE, rep)
        self.gdp_weights = gdp_weights
        self.rebalancing_thresholds = rebalancing_thresholds
        self.trading_costs = trading_costs
        self.code = f'liqid_global_var'
        self._equities_regions = ['North America', 'Europe', 'Pacific ex Japan', 'Japan', 'Emerging Markets']
        self._equities_dm = ['North America', 'Europe', 'Pacific ex Japan', 'Japan']
        self._equities_em = ['Emerging Markets']
        self._bonds = ['Bonds EUR Govt', 'Emerging Market Bonds World', 'High Yield World', 'Inflation Linked Bonds World', 'Investment Grade Corporate Bonds World']
        self._commodities = ['Gold', 'Commodities_ex_Gold', 'Private Equity', 'Real Estate Participations']
        self._risky_assets = self._equities_regions + ['Private Equity'] + ['Real Estate Participations']
        self._safe_assets = self._bonds + ['Gold'] + ['Commodities_ex_Gold']
        self._sub_asset_classes = self._equities_regions + self._bonds + self._commodities
        self._sub_asset_rebalancing = ['sub_GOLD', 'sub_COM', 'sub_BONDS_IG', 'sub_BONDS_HY', 'sub_EQU_DM', 'sub_EQU_EM']
        self.portfolio_return = pd.DataFrame(columns=[self.code])
        self.rebalancing_dates = []
        self.first_of_july_list = []  # get a list of the first trading day of July for annual rebalancing
        for i, day in enumerate(asset_returns.index[1:]):
            if (day.month != asset_returns.index[i].month) and day.month == 1:
                self.first_of_july_list.append(day)
        self.daily_fee_factor = (1 - fees) ** (1 / 252)

        self.get_target_weights(day=self.asset_returns.index[0])
        self._bonds_commodities_target_weights = self.weights_assets[self._bonds + self._commodities].copy()

        self.calculate()
        self.get_indexed_return_series()

    def get_target_weights(self, day):
        # Passt die GDP weights auf die jeweilige Risikoklasse an. Und gewichtet Bonds und Equities.
        # Bei Bonds noch 5% in high yields
        # Fields ist eins von Gold, Commodities, Equities, Bonds
        # Jeden Tag aufgerufen
        risk_name = ['Equities', 'Bonds', 'Gold', 'Commodities_ex_Gold', 'Private Equity', 'Real Estate Participations']
        self.weights_asset_class_level = pd.DataFrame(self.risk_class, index=risk_name, columns=[self.asset_returns.index[0]]).T  # day start weights for all DataFrames

        ### Das hier kann platzeffektiver geschrieben werden
        na = self.gdp_weights['North America'].values[0]
        eu = self.gdp_weights['Europe'].values[0]
        pexj = self.gdp_weights['Pacific ex Japan'].values[0]
        jpn = self.gdp_weights['Japan'].values[0]
        em = self.gdp_weights['Emerging Markets'].values[0]
        BEURGov = self.gdp_weights['Bonds EUR Govt'].values[0]
        embw = self.gdp_weights['Emerging Market Bonds World'].values[0]
        hyw = self.gdp_weights['High Yield World'].values[0]
        ilbw = self.gdp_weights['Inflation Linked Bonds World'].values[0]
        igcbw = self.gdp_weights['Investment Grade Corporate Bonds World'].values[0]
        g = self.gdp_weights['Gold'].values[0]
        comm = self.gdp_weights['Commodities_ex_Gold'].values[0]
        pe = self.gdp_weights['Private Equity'].values[0]
        rep = self.gdp_weights['Real Estate Participations'].values[0]
        self.weights_sub_asset_class_level = pd.DataFrame([na, eu, pexj, jpn, em, BEURGov, embw, hyw, ilbw, igcbw, g, comm, pe, rep], index=self._sub_asset_classes, columns=[day]).T #Reihenfolge kann oben bei Def von self._sub_asset_class nachgeschaut werden

        self.weights_assets = pd.DataFrame()
        self.weights_assets = self.weights_assets.append(self.weights_sub_asset_class_level[self._equities_regions + self._bonds + self._commodities].T).T


    def calculate(self):
        for i, date in enumerate(self.asset_returns.index[:-1]):  # geht durch Index bis auf letzte Zeile
            next_day = self.asset_returns.index[i + 1]  # nächster Tag ist Index des Datums
            return_ = sum(self.asset_returns.loc[date] * self.weights_assets.loc[date])  # aufsummierter return auf PF-create_database_connection
            self.portfolio_return = self.portfolio_return.append(pd.DataFrame(return_, columns=[self.code], index=[date]))
            self.portfolio_return.iloc[-1, :] = (1 + self.portfolio_return.iloc[-1, :]) * self.daily_fee_factor - 1  # falsch, quartalsweise
            self.weights_assets.loc[next_day, :] = ((1+self.asset_returns.loc[date, :]) * self.weights_assets.loc[date, :]) / (1 + return_)

            self.save_weights_rebalancing_units(next_day)
            self.check_rebalancing(next_day)

    def save_weights_rebalancing_units(self, day):
        self.weights_asset_class_level.loc[day, 'Equities'] = \
        self.weights_assets[self._equities_regions].sum(axis=1).loc[day]
        self.weights_asset_class_level.loc[day, 'Bonds'] = self.weights_assets[self._bonds].sum(axis=1).loc[day]
        self.weights_asset_class_level.loc[day, ['Gold']] = self.weights_assets[['Gold']].loc[day]
        self.weights_asset_class_level.loc[day, ['Commodities_ex_Gold']] = self.weights_assets[['Commodities_ex_Gold']].loc[day]
        self.weights_asset_class_level.loc[day, ['Private Equity']] = self.weights_assets[['Private Equity']].loc[day]
        self.weights_asset_class_level.loc[day, ['Real Estate Participations']] = self.weights_assets[['Real Estate Participations']].loc[day]
        self.weights_sub_asset_class_level.loc[day, self._equities_dm + self._equities_em + self._bonds + self._commodities] = self.weights_assets[self._equities_dm +self._equities_em + self._bonds + self._commodities].loc[day]


    def check_rebalancing(self, day):
        ### Soll wirklich jährlich rebalanct werden? Was ergibt es für ein Sinn, wenn man sowieso nach weights checkt?...
        # Detektiert Daten für Rebalancing und gibt "Gründe" dafür an, die bei rebalancing dokumentiert werden
        # target_loc = self.gdp_weights.index.get_loc(day, 'pad')
        # check annual rebalancing in June. (Not in June, but beginning of year now)

        if day in self.first_of_july_list:
            self.rebalance(day, trigger='Annual Rebalancing')
        # check threshold on asset class level
        elif abs(self.weights_assets.loc[day, self._risky_assets].sum() - (1 - self.gdp_weights[self._safe_assets].iloc[0].sum())) > self.rebalancing_thresholds['Asset Class Level']['Risky Assets']:
            self.rebalance(day, trigger='Asset Class Level')

        # check threshold on sub asset class level
        else:
            for sub_asset_class in self._sub_asset_rebalancing:
                if sub_asset_class == 'sub_GOLD':
                    assets = ['Gold']
                elif sub_asset_class == 'sub_COM':
                    assets = ['Commodities_ex_Gold']
                elif sub_asset_class == 'sub_BONDS_IG':
                    assets = ['Bonds EUR Govt', 'Inflation Linked Bonds World', 'Investment Grade Corporate Bonds World']
                elif sub_asset_class == 'sub_BONDS_HY':
                    assets = ['Emerging Market Bonds World', 'High Yield World']
                elif sub_asset_class == 'sub_EQU_DM':
                    assets = ['North America', 'Europe', 'Japan', 'Private Equity', 'Real Estate Participations']
                elif sub_asset_class == 'sub_EQU_EM':
                    assets = ['Pacific ex Japan', 'Emerging Markets']
                else:
                    assets = []
                # Abweichung ist jetziger Stand minus dem Anfangswert
                dev = abs(self.weights_sub_asset_class_level[assets].loc[day].sum() - self.gdp_weights[assets].iloc[0].sum())
                if dev >= self.rebalancing_thresholds['Sub Asset Class Level'][sub_asset_class]:
                    self.rebalance(day, trigger='Sub Asset Class Level')
            for region, threshold in self.rebalancing_thresholds['Region'].items():
                dev = abs(self.weights_assets.loc[day, region] - self.gdp_weights.iloc[0][region])
                if dev >= threshold:
                    self.rebalance(day, trigger=f'Region Level: {region}')

    def rebalance(self, day, trigger):

        print(trigger)
        # Dokumentiert rebalancing dates mit Grund
        # Setzt alle Weights auf Ausgangspunkt zurück und dokumentiert das trading volume
        # Aus Trading Volumes werden Tradingkosten berechnet und vom Wert des Portfolios abgezogen.
        # The rebalancing occurs on the day the threshold is reached. Therefore, the responsible line is not visible weights_sub_asset_level, since it will be overwritten by rebalance().
        self.rebalancing_dates.append((day, trigger))
        weights_before_rebalancing = self.weights_assets.loc[day, :].copy()
        target_loc = self.gdp_weights.index.get_loc(day, 'pad')
        self.weights_assets.loc[day, self._equities_regions] = self.gdp_weights.iloc[target_loc, :][
                                                                   self._equities_regions]
        self.weights_assets.loc[day, self._bonds] = self._bonds_commodities_target_weights[self._bonds].values
        self.weights_assets.loc[day, self._commodities] = self._bonds_commodities_target_weights[
            self._commodities].values

        volume_traded = sum(abs(self.weights_assets.loc[day, :] - weights_before_rebalancing))
        trading_costs = volume_traded * self.trading_costs
        self.portfolio_return.iloc[-1, :] = self.portfolio_return.iloc[-1, :] - trading_costs

        self.save_weights_rebalancing_units(day)

    def get_indexed_return_series(self):
        date = self.asset_returns.index[0] + pd.Timedelta(days=-1)
        self.indexed_returns = pd.DataFrame(0, index=[date], columns=[self.code])
        self.indexed_returns = self.indexed_returns.append(self.portfolio_return)
        self.indexed_returns = self.indexed_returns.add(1).cumprod()
        self.indexed_returns = self.indexed_returns.resample('M').last()
        self.indexed_returns = self.indexed_returns.reset_index()
        self.indexed_returns['model_portfolio_key'] = self.code
        self.indexed_returns = self.indexed_returns.rename(columns={'index': 'date', self.code: 'quote'})[['model_portfolio_key', 'date', 'quote']]


def main(rebalancing_thresholds):
    ### Check again, if correct values were used
    product_cost_dict = {'Bonds EUR Govt': (0.0017),
                         'Emerging Market Bonds World': (0.0045),
                         'High Yield World': (0.0005),
                         'Inflation Linked Bonds World': (0.002),
                         'Investment Grade Corporate Bonds World': (0.0025),
                         'Pacific ex Japan': (0.006),
                         'Emerging Markets': (0.0018),
                         'Europe': (0.0012),
                         'Japan': (0.0012),
                         'North America': (0.0007),
                         'Commodities_ex_Gold': (0.0019),
                         'Gold': (0.0000),
                         'Private Equity': (0.0075),
                         'Real Estate Participations': (0.0025)
                         }

    path_gdp_weights = r"Summary_GDP_Weights Equities Liqid.xlsx"
    gdp_weights = prepare_gdp_weights(path_gdp_weights)

    # Entfernen von Risk class: Ziel: Nur eine Risikoklasse simulieren
    '''
    Risk_Class = namedtuple('RiskClass', ['Equities', 'Bonds', 'Gold', 'Commodities_ex_Gold'])
    risk_classes = [Risk_Class(x[0] / 100, x[1] / 100, 0.02, 0.03) for x in zip(range(5, 110, 10), range(90, -10, -10))]
    '''

    returns = load_time_series_bloomberg()
    returns = deduct_costs(returns, product_cost_dict)

    all_data = pd.DataFrame()
    portfolios = []

    port = Portfolio(returns, gdp_weights, rebalancing_thresholds, trading_costs=0.002, fees=0.005)  # Causes error
    portfolios.append(port)
    all_data = all_data.append(port.indexed_returns)
    all_data.to_csv(r'Ergebnisse\liqid_global_simulations.csv', index=False)
    print('Liqid Global Simulation finished')
    return all_data


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Bald entfernen und Pandas append function sollte durch concat ersetzen

###Weights sind inkonsitent und müssen alle überprüft und verbessert werden
rebalancing_threshold_list = [{'Asset Class Level': {'Risky Assets': 0.05},
                               'Sub Asset Class Level': {'sub_GOLD': 0.01,
                                                         'sub_COM': 0.01,
                                                         'sub_BONDS_IG': 0.0425,
                                                         'sub_BONDS_HY': 0.005,
                                                         'sub_EQU_DM': 0.0425,
                                                         'sub_EQU_EM': 0.0075},
                               'Region': {'North America': 0.018*monte_count/1000,
                                          'Europe': 0.007*monte_count/1000,
                                          'Japan': 0.002*monte_count/1000,
                                          'Emerging Markets': 0.023*monte_count/1000,
                                          'Pacific ex Japan': 0.007*monte_count/1000}} for monte_count in range(8, 8000, 8)] #(1, 2000, 2)

graph_data = pd.DataFrame()
for rebalancing_thresholds in rebalancing_threshold_list:
    iteration_result = main(rebalancing_thresholds)
    graph_data[str(rebalancing_thresholds['Region']['North America'])] = iteration_result['quote']
graph_data.index = iteration_result['date']
graph_data.to_csv(r'Ergebnisse\thresholds.csv', index=True)
graph_data.plot()
plt.show()

'''
graph_data = pd.DataFrame()
for rebalancing_thresholds in rebalancing_threshold_list:
    result_data = main(rebalancing_thresholds)
    graph_data = graph_data.append(result_data['quote'])
graph_data.T.to_csv(r'Ergebnisse\thresholds.csv', index=False)
graph_data.T.plot()
plt.show()'''