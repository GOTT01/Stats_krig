#!/usr/bin/env python
# -*- coding: utf-8 -*-

#==============================================================================
# Augusto Oliveira
# augoliv@gmail.com
#==============================================================================

import numpy as np

def outlier_trimmer(df, datacolumn, peso=1.5):
    n = df.shape[0]
    df_sample = df.sort_values(datacolumn)
    data = df_sample[datacolumn]
    sorted(data)
    Q1,Q3 = np.percentile(data , [25,75])
    IQR = Q3 - Q1
    l = Q1 - (peso * IQR)
    u = Q3 + (peso * IQR)
    df_sample_trimmed = df_sample.loc[df_sample[datacolumn].between(l,u)]
    p = df_sample_trimmed.shape[0]
    return df_sample_trimmed, n-p


def calcula_mape(predicao, real): 
    mape = np.mean(np.abs((real - predicao) / real)) * 100
    return mape

# Função para calcular Root Mean Squared Logarithmic Error (RMSLE)
def calcula_rmsle(predicao, real):
    from sklearn.metrics import mean_squared_log_error
    return np.sqrt(mean_squared_log_error(y_true=real, y_pred=predicao))


# Função para calcular a MEDIANA do nível de avaliação de cada predição
def calcula_Mediana_Sales_Ratio(predicao, real):
    ASR_median = np.median(predicao/real)
    return ASR_median

# Função para calcular o COD - Coeficiente de Dispersão
def calcula_Coeficiente_Dispersao(predicao, real):
    '''
        1. subtract the median from each ratio
        2. take the absolute value of the calculated differences
        3. sum the absolute differences
        4. divide by the number of ratios to obtain the average absolute deviation
        5. divide by the median
        6. multiply by 100
    '''
    predicao = predicao.ravel()
    real = real.ravel()
    # Calcule a mediana
    SR_mediano = np.median(predicao/real)
    # Calcule o sales ratio para cada ponto (gera um array)
    SR = predicao/real
    # Subtraia a mediana de cada ratio valor absoluto (gera um array)
    SR_minus_median_abs = np.abs(SR - SR_mediano)
    # Tire a média acima (gera um escalar)
    SR_minus_median_abs_mean = np.mean(SR_minus_median_abs)
    # Divida pela mediana (gera um escalar)
    avg_abs_dev_divide_by_median = SR_minus_median_abs_mean/SR_mediano
    # Multiplique por 100 (gera um escalar)
    COD = avg_abs_dev_divide_by_median * 100.0
    return COD


def calcula_PRD(predicao, real):
    predicao = predicao.ravel()
    real = real.ravel()
    ASR = predicao / real
    MeanRatio = np.mean(ASR)
    MedianRatio = np.median(predicao / real)
    TotalOfAssessedValues = np.sum(predicao)
    TotalOfSalesPrices = np.sum(real)
    WeightedMean = TotalOfAssessedValues / TotalOfSalesPrices
    PRD = MeanRatio / WeightedMean

    if PRD < 0.98:
        status = 'Progressividade'
    else:
        if PRD > 1.03:
            status = 'Regressividade'
        else:
            status = 'Normal'

    return {'PRD': PRD, 'Status': status}


def calcula_PRB(predicao, real):
    '''
    Low- or high-value properties are appraised at equal percentages of market value?
    
    It can be interpreted as the expected change in ratios as property values
double. If ratios increase as property values increase, the resulting PRB
will be positive. 
    For example, a PRB of 0.025 indicates that if property
value doubles, ratios increase by 2.5 percent. 
    A positive PRB indicates that assessments are progressive,
meaning high-value properties are over-appraised relative to low-value
properties (and the opposite for an equivalent negative coefficient).

    The PRB provides a measure of price-related bias that is
more meaningful and less sensitive to extreme prices or ratios. 
    As a general matter, the PRB coefficient should fall between –0.05 and 0.05. 
    PRBs for which 95% confidence intervals fall outside of this range indicate that one can
reasonably conclude that assessment levels change by more than 5% when values 
are halved or doubled. 
    PRBs for which 95% confidence intervals fall outside the range
of –0.10 to 0.10 indicate unacceptable vertical inequities.
    '''
    import statsmodels.api as sm

    predicao = predicao.ravel()
    real = real.ravel()
    ASR = predicao / real
    MedianRatio = np.median(predicao / real)

    Pct_Diff = (ASR - MedianRatio) / MedianRatio

    values = 0.5 * real + 0.5 * predicao / MedianRatio
    indepvar = np.log(values) / 0.693
    depvar = Pct_Diff

    X = sm.add_constant(indepvar)
    result = sm.OLS(depvar, X).fit()

    PRB = result.params[1]
    PRB_inf = result.conf_int(alpha=0.05, cols=[1])[0][0]
    PRB_sup = result.conf_int(alpha=0.05, cols=[1])[0][1]

    status = 'Erro'
    if PRB_inf * PRB_sup >= 0:
        if abs(PRB_inf)<0.10 and abs(PRB_sup)>0.10:
            status = 'Inequidade vertical inaceitável com intervalo superior a +/- 10%'            
        else:        
            if PRB >= 0 and PRB <= 0.05:
                status = 'Progressividade normal dentro do intervalo de +/- 5%'
            elif PRB > 0.05 and PRB <= 0.10:
                status = 'Progressividade dentro do intervalo de +/- 10%'
            elif PRB > 0.10:
                status = 'Progressividade maior que 10%'
                
            elif PRB < 0 and PRB >= -0.05:
                status = 'Regressividade normal dentro do intervalo de +/- 5%'
            elif PRB < -0.05 and PRB >= -0.10:
                status = 'Regressividade dentro do intervalo de +/- 10%'
            elif PRB < -0.10:
                status = 'Regressividade menor que -10%'
    else:
        status = 'PRB apresentou valor não conclusivo no teste de significância.'

    return {'PRB': PRB, 'PRB_inf': PRB_inf, 'PRB_sup': PRB_sup, 'Status': status}



def calcula_metricas(predicao, real, titulo='', imprimir=False, trimm = False, peso=1.5):
    from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error   
    import pandas as pd
    
    n = len(predicao)
    n_exc = 0
    if trimm:
        df = pd.DataFrame({'predicao': predicao, 'real': real})
        df['ASR'] = df['predicao']/df['real']
        df, n_exc = outlier_trimmer(df, 'ASR', peso=peso)
        asr = df['ASR']
        predicao = df['predicao']
        real = df['real']


    asr = calcula_Mediana_Sales_Ratio(predicao=predicao, real=real)

    cod = calcula_Coeficiente_Dispersao(predicao = predicao, real = real)

    prd = calcula_PRD(predicao=predicao, real=real)
    PRD = prd['PRD']
    status_PRD = prd['Status']

    prb = calcula_PRB(predicao=predicao, real=real)
    PRB = prb['PRB']
    status_PRB = prb['Status']

    rmse = np.sqrt(mean_squared_error(predicao, real))
    
    rmsle_ = calcula_rmsle(real, predicao)
    
    mae = mean_absolute_error(y_true = real, y_pred = predicao)

    mape = calcula_mape(predicao = predicao, real = real)

    r2score = r2_score(y_true = real, y_pred = predicao)

    df_res = pd.DataFrame(columns=['nivel_avaliacao','cod','rmse','rmsle','mae','mape','r2', 'prd', 'prb', 'status_prd', 'status_prb'])
    data = {'titulo': titulo,
            'n': n, 
            'n_exc':n_exc,
            'trimm':trimm,
            'peso':peso,
            'nivel_avaliacao':asr, 
            'cod':cod, 
            'rmse':rmse, 
            'rmsle':rmsle_, 
            'mae':mae,
            'mape':mape, 
            'r2':r2score,
            'prd':PRD, 
            'prb':PRB, 
            'status_prd':status_PRD, 
            'status_prb':status_PRB
            }
    df_res = df_res.append(data, ignore_index=True)

    if imprimir:
        
        if titulo!='':
            print("======================================================")
            print(titulo)
            print("======================================================")
        
        print('Número de dados:                             %8.0f' % n)
        if trimm:
            print('Número de dados excluídos (outliers trimm):  %8.0f' % n_exc)       
        print('Nível de avaliação (Sales_ratio_median):     %8.2f' % asr)
        print('Coeficiente de dispersão (COD):              %8.2f%%' % cod)
        print('Price-Related Differential (PRD):            %9.3f (%s)' % (PRD, status_PRD))
        print('Price-Related Bias (PRB):                    %9.3f (%s)' % (PRB, status_PRB))
        print("Root Mean Squared Error (RMSE):              %8.2f" % (rmse))
        print("Root Mean Squared Logarithmic Error (RMSLE): %8.2f" % (rmsle_))
        print("Mean Absolute Error (MAE):                   %8.2f" % mae)
        print('Mean absolute percentage error (MAPE):       %8.2f%%' % mape)
        print("R² Score:                                    %8.2f" % r2score)
        if titulo == '':
            print("------------------------------------------------------")
        else:
            print('\n')
    
    return df_res


def calcula_metricas_IAAO(predicao, real):
    calcula_Mediana_Sales_Ratio(predicao, real)
    calcula_Coeficiente_Dispersao(predicao, real)


# Função para calcular e retorna toda a planilha do IAAO
def DataFrame_IAAO(predicao, real):
    import pandas as pd
    from scipy import stats    
    
    # Calcula a média do Sales Ratio
    ASR_mean = np.mean(predicao/real)
    
    # Weighted  mean 
    weighted_mean = np.mean(predicao)/ np.mean(real)
    
    # PRD The mean divided by the weighted mean. The statistic has a slight bias upward.
    '''
        Price-related differentials above 1.03 tend to indicate assessment regressivity; 
        price-related differentials below 0.98 tend to indicate assessment progressivity.
    '''
    PRD = ASR_mean/weighted_mean
    
    # Inicializa o DataFrame
    df_IAAO = pd.DataFrame({'AV': predicao, 'SP': real})
    
    # Calcule o sales ratio para cada ponto (gera um array)
    SR = predicao/real  
    
    df_IAAO['ASR'] = SR
    
    # Calcula a mediana de Sales Ratio
    ASR_median = calcula_Mediana_Sales_Ratio(predicao, real)
    
    # Colunas da planilha do IAAO
    df_IAAO['AV/Median'] = df_IAAO['AV']/ASR_median
    df_IAAO['Value'] = 0.5*df_IAAO['SP'] + 0.5*df_IAAO['AV/Median']
    df_IAAO['lndep_Var'] = np.log(df_IAAO['Value'])/0.693
    df_IAAO['Dep_Variable'] = (df_IAAO['ASR']-ASR_median)/ASR_median
    
    # Calcula o coeficiente de dispersão
    COD = calcula_Coeficiente_Dispersao(predicao, real)
    
    # Calcula a reta de regressão para o PRB
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_IAAO['lndep_Var'],df_IAAO['Dep_Variable'])
    
    t_value = slope/std_err
    
    print('Price-related differentials (PRD): %10.4f' % PRD)
    
    if (PRD > 1.03):
        print('Regressividade detectada (PRD > 1,03)!')
    elif (PRD < 0.98):
        print('Progressividade (PRD < 0.98)!')
    else:
        print('Progressividade/Regressividade OK (0,98 < PRD < 1,03)!')
        
    print('Price Related Bias (PRB): %10.4f' % slope)
    print('Std Error: %10.4f' % std_err)
    print('t_value: %10.4f' % t_value)
    print('Sig. F Snedocor: %10.4f' % p_value)
    
   
    return df_IAAO
