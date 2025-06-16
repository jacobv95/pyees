
from testData.windTunnelTest.computeAirFlow import computeAirFlow
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
import numpy as np
from variable import variable
from sheet import sheetsFromFile, fileFromSheets
from fit import *
from prop import prop

## class to define a test of a cooler
class TestOfCooler:
    def __init__(self, number, dataFiles, nNozzles):
        self.number = number
        self.dataFiles = dataFiles
        self.nNozzles = nNozzles
        

## list of tests of coolers
tests = [
    TestOfCooler(
        number=3,
        dataFiles=["20250520_075235.xlsx",
                   "20250526_075851.xlsx" ],
        nNozzles=[2,
                  2]
        )
]

nAirFlows = 5

figScpLiq, axScpLiq = plt.subplots()
figScpAir, axScpAir = plt.subplots()

for test in tests:
    
    ## append the datafiles and include the number of nozzles used
    for i, (dataFile, nNozzle) in enumerate(zip(test.dataFiles, test.nNozzles)):
        if i == 0:
            data = sheetsFromFile(f"testData/windTunnelTest/{test.number}/{dataFile}", dataRange = "A-L", uncertRange = "M-X")
            data.nNozzles = variable([nNozzle] * len(data))
        else:
            dat = sheetsFromFile(f"testData/windTunnelTest/{test.number}/{dataFile}", dataRange = "A-L", uncertRange = "M-X")
            dat.nNozzles = variable([nNozzle] * len(dat))
            data.append(dat)
    

    ## the measured relative humidity was below zero. Cap the value.
    data.rh_air = variable([np.max([0.0, elem]) for elem in data.rh_air.value] , data.rh_air.unit)

    ## select the air side temperatures to use
    data.t_air_in = data.t_air_in_pt100
    data.t_air_out = data.t_air_out_pt100


    ## calculate the air flow
    d_nozzle = variable(225, 'mm')
    # data.flow_air = computeAirFlow(
    #     n_nozzle=data.nNozzles, 
    #     d_nozzle=d_nozzle, 
    #     dp_nozzle=data.dP_nozzle, 
    #     p_atm=data.p_atm,
    #     T_5 = data.t_air_out, 
    #     T_6=data.t_air_out, 
    #     P_s5=data.p_atm - data.dP_air,
    #     P_s6=data.p_atm - data.dP_air - data.dP_nozzle,
    #     rh_5 = data.rh_air,
    #     rh_6 = data.rh_air,
    #     C_initial = 1
    # )
    data.flow_air = variable(list(range(1,31)), 'm3/h')

    ## air side material properties
    data.absolute_humitidy = prop('humidity', 'air', t =  data.t_air_out, p = data.p_atm - data.dP_air, rh = data.rh_air)
    data.rho_air = prop('density',       'air',    t = (data.t_air_in + data.t_air_out) / 2,     p = data.p_atm - data.dP_air / 2,   humidity = data.absolute_humitidy)
    data.cp_air  = prop('specific_heat', 'air',    t = (data.t_air_in + data.t_air_out) / 2,     p = data.p_atm - data.dP_air / 2,   humidity = data.absolute_humitidy)

    ## liquid side material properties
    data.rho_liq = prop('density',       'MEG',  t = (data.t_liq_in + data.t_liq_out) / 2,     p = variable(1, 'bar'),  C = variable(33, '%'))
    data.cp_liq  = prop('specific_heat', 'MEG',  t = (data.t_liq_in + data.t_liq_out) / 2,     p = variable(1, 'bar'),  C = variable(33, '%'))

    ## performance
    data.phi_liq = data.rho_liq * data.flow_liq * data.cp_liq * (data.t_liq_in - data.t_liq_out)
    data.phi_air = data.rho_air * data.flow_air * data.cp_air * (data.t_air_out - data.t_air_in)
    data.phi_liq.convert('kW')
    data.phi_air.convert('kW')

    ## specific performance
    data.scp_liq = data.phi_liq / (data.t_liq_in - data.t_air_in)
    data.scp_air = data.phi_air / (data.t_liq_in - data.t_air_in)

    ## performance ratio
    data.r_phi = (data.phi_liq - data.phi_air) / data.phi_liq
    data.r_phi.convert('%')


    ## partition the dataset
    labels = []
    dataSets = []
    airFlows = []
    for i in range(int(len(data) / nAirFlows)):
        dat = data[i*nAirFlows : i*nAirFlows + nAirFlows]
        dataSets.append(dat)

        label = np.mean(dat.flow_air)
        airFlows.append(label)
        labels.append(f"{label.value:.2f} {label.unit}")
    indexes = np.argsort(airFlows)
    indexes = list(reversed(indexes))
    labels = [labels[i] for i in indexes]
    dataSets = [dataSets[i] for i in indexes]

    ## Peformance
    fig, ax = plt.subplots()
    ax.set_xlabel('Liquid flow')
    ax.set_ylabel('Peformance')

    for dat, lab, color in zip(dataSets, labels, colors):
        f = pow_fit(dat.flow_liq / variable(1, dat.flow_liq.unit), dat.phi_liq)
        f.plot(ax, color = color, label = lab)

        f = dummy_fit(dat.flow_liq, dat.phi_liq)
        f.scatter(ax, color = color, label = None)

        f = pow_fit(dat.flow_liq / variable(1, dat.flow_liq.unit), dat.phi_air)
        f.plot(ax, color = color, label = None, linestyle = 'dashed')

        f = dummy_fit(dat.flow_liq, dat.phi_air)
        f.scatter(ax, color = color, label = None)

    f.addUnitToLabels(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'testData/windTunnelTest/{test.number}/Performance liquid flow.png')



    ## Performance ratio
    fig, ax = plt.subplots()
    ax.set_xlabel('Liquid flow')
    ax.set_ylabel('Peformance ratio')


    for dat, lab, color in zip(dataSets, labels, colors):    
        f = dummy_fit(dat.flow_liq, dat.r_phi)
        f.scatter(ax, color = color, label = None)
        f.plotData(ax, label = lab)

    f.addUnitToLabels(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'testData/windTunnelTest/{test.number}/Performance ratio liquid flow.png')



    ## scp
    fig, ax = plt.subplots()
    ax.set_xlabel('Liquid flow')
    ax.set_ylabel('Speicifc performance')
    axScpLiq.set_xlabel('Liquid flow')
    axScpLiq.set_ylabel('Speicifc performance')

    for dat, lab, color in zip(dataSets, labels, colors):
        f = pow_fit(dat.flow_liq / variable(1, dat.flow_liq.unit), dat.scp_air)
        f.plot(ax, label = None, color = color, alpha = 0.3)

        f = dummy_fit(dat.flow_liq, dat.scp_air)
        f.scatter(ax, label = None, color = color, alpha = 0.3)

        f = pow_fit(dat.flow_liq / variable(1, dat.flow_liq.unit), dat.scp_liq)
        f.plot(ax, label = lab, color = color)
        f.plot(axScpLiq, label = f'Cooler {test.number} {lab}', color = color)
        f.plotUncertanty(ax, label = None, color = color)
        
        f = dummy_fit(dat.flow_liq, dat.scp_liq)
        f.scatter(ax, label = None, color = color)
        f.scatter(axScpLiq, label = None, color = color)

    f.addUnitToLabels(ax)
    f.addUnitToLabels(axScpLiq)

    ax.legend()
    fig.tight_layout()
    fig.savefig(f'testData/windTunnelTest/{test.number}/specific performance liquid flow.png')




    ## partition the dataset
    labels = []
    dataSets = []
    liquidFlows = []
    n = int(len(data) / nAirFlows)
    m = int(len(data) / n)
    for i in range(m):
        indexes = [ii * nAirFlows + i for ii in range(n)]
        dat = data[indexes]
        dataSets.append(dat)

        label = np.mean(dat.flow_liq)
        liquidFlows.append(label)
        labels.append(f"{label.value:.2f} {label.unit}")
    indexes = np.argsort(liquidFlows)
    indexes = list(reversed(indexes))
    labels = [labels[i] for i in indexes]
    dataSets = [dataSets[i] for i in indexes]


    ## Peformance
    fig, ax = plt.subplots()
    ax.set_xlabel('Air flow')
    ax.set_ylabel('Peformance')

    for dat, lab, color in zip(dataSets, labels, colors):
        f = pow_fit(dat.flow_air / variable(1, dat.flow_air.unit), dat.phi_liq)
        f.plot(ax, color = color, label = lab)

        f = dummy_fit(dat.flow_air, dat.phi_liq)
        f.scatter(ax, color = color, label = None)

        f = pow_fit(dat.flow_air / variable(1, dat.flow_air.unit), dat.phi_air)
        f.plot(ax, color = color, label = None, linestyle = 'dashed')

        f = dummy_fit(dat.flow_air, dat.phi_air)
        f.scatter(ax, color = color, label = None)

    f.addUnitToLabels(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'testData/windTunnelTest/{test.number}/Performance air flow.png')



    ## Performance ratio
    fig, ax = plt.subplots()
    ax.set_xlabel('Air flow')
    ax.set_ylabel('Peformance ratio')


    for dat, lab, color in zip(dataSets, labels, colors):
        f = dummy_fit(dat.flow_air, dat.r_phi)
        f.scatter(ax, color = color, label = None)
        f.plotData(ax, label = lab)

    f.addUnitToLabels(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'testData/windTunnelTest/{test.number}/Performance ratio air flow.png')



    ## scp
    fig, ax = plt.subplots()
    ax.set_xlabel('Air flow')
    ax.set_ylabel('Speicifc performance')
    axScpAir.set_xlabel('Air flow')
    axScpAir.set_ylabel('Speicifc performance')

    for dat, lab, color in zip(dataSets, labels, colors):
        f = pow_fit(dat.flow_air / variable(1, dat.flow_air.unit), dat.scp_air)
        f.plot(ax, label = None, color = color, alpha = 0.3)

        f = dummy_fit(dat.flow_air, dat.scp_air)
        f.scatter(ax, label = None, color = color, alpha = 0.3)

        f = pow_fit(dat.flow_air / variable(1, dat.flow_air.unit), dat.scp_liq)
        f.plot(ax, label = lab, color = color)
        f.plot(axScpAir, label = f'Cooler {test.number} {lab}', color = color)
        f.plotUncertanty(ax, label = None, color = color)
        
        
        f = dummy_fit(dat.flow_air, dat.scp_liq)
        f.scatter(ax, label = None, color = color)
        f.scatter(axScpAir, label = None, color = color)

    f.addUnitToLabels(ax)
    f.addUnitToLabels(axScpAir)

    ax.legend()
    fig.tight_layout()
    fig.savefig(f'testData/windTunnelTest/{test.number}/specific performance air flow.png')





    # save calculations
    fileFromSheets(data, f'testData/windTunnelTest/{test.number}/calculatedValues.xlsx')
    for meas in data:
        for elem in meas:
            elem._uncert = 0
    fileFromSheets(data, f'testData/windTunnelTest/{test.number}/calculatedValuesWithoutUncertanty.xlsx')




axScpLiq.legend()
figScpLiq.tight_layout()
figScpLiq.savefig(f'testData/windTunnelTest/specific performance liquid flow.png')

axScpAir.legend()
figScpAir.tight_layout()
figScpAir.savefig(f'testData/windTunnelTest/specific performance air flow.png')

