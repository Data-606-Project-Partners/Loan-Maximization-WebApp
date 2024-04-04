import pandas as pd;
import numpy as np;

def convertEmploymentLength(number):
    if int(number) < 1:
        return ( '< 1 year' )
    elif int(number) == 1:
        return( '1 year' )
    elif int(number) >= 10:
        return ( '10+ years' )
    else:
        return(f'{number} years')

def transform(data, LabelEncodingModel, ClassificationModel):

    #print(ClassificationModel)
    #print(ClassificationModel.feature_names_in_)

    transformedDF = pd.DataFrame(np.zeros(shape=(1,len(ClassificationModel.feature_names_in_))), columns = ClassificationModel.feature_names_in_)

    transformedDF['funded_amnt'] = transformedDF['funded_amnt'].astype(int)
    transformedDF['term'] = transformedDF['term'].astype(int)
    transformedDF['issue_year'] = transformedDF['issue_year'].astype(int)
    transformedDF['issue_month'] = transformedDF['issue_month'].astype(int)
    transformedDF['revol_bal'] = transformedDF['revol_bal'].astype(int)

    transformedDF['has_delinq'] = transformedDF['has_delinq'].astype(bool)
    transformedDF['has_record'] = transformedDF['has_record'].astype(bool)
    transformedDF['has_derog'] = transformedDF['has_derog'].astype(bool) 

    transformedDF['sub_grade'] = transformedDF['sub_grade'].astype(object)
    transformedDF['emp_length'] = transformedDF['emp_length'].astype(object)
    transformedDF['home_ownership'] = transformedDF['home_ownership'].astype(object)
    transformedDF['verification_status'] = transformedDF['verification_status'].astype(object)
    transformedDF['purpose'] = transformedDF['purpose'].astype(object)
    transformedDF['addr_state'] = transformedDF['addr_state'].astype(object)
    transformedDF['initial_list_status'] = transformedDF['initial_list_status'].astype(object)
    transformedDF['zip_code'] = transformedDF['zip_code'].astype(object)

    # Ints
    transformedDF.loc[0, 'funded_amnt'] = int(data['funded_amnt'][:-3])
    transformedDF.loc[0, 'term'] = pd.to_numeric(data['term'][0:2])
    transformedDF.loc[0, 'issue_year'] = int(data['issue_d'][0:4])
    transformedDF.loc[0, 'issue_month'] = int(data['issue_d'][-2:])
    transformedDF.loc[0, 'revol_bal'] = int(data['revol_bal'][:-3])

    # Bools
    transformedDF.loc[0, 'has_delinq'] = data['has_delinq']
    transformedDF.loc[0, 'has_record'] = data['has_record']
    transformedDF.loc[0, 'has_derog'] = data['has_derog']

    # Objects (Strings)
    transformedDF.loc[0, 'sub_grade'] = data['sub_grade']
    transformedDF.loc[0, 'emp_length'] = convertEmploymentLength(data['emp_length'])
    transformedDF.loc[0, 'home_ownership'] = data['home_ownership']
    transformedDF.loc[0, 'verification_status'] = data['verification_status']
    transformedDF.loc[0, 'purpose'] = data['purpose']
    transformedDF.loc[0, 'addr_state'] = data['addr_state']
    transformedDF.loc[0, 'initial_list_status'] = data['initial_list_status']
    transformedDF.loc[0, 'zip_code'] = data['zip_code'][:-2]+'xx'

    # Floats (order matters because otherwise information is undefined at that point)
    transformedDF.loc[0, 'installment'] = float(data['installment'])
    transformedDF.loc[0, 'annual_inc'] = max(float(1),float(data['annual_inc']))
    transformedDF.loc[0, 'installment_to_income_ratio'] = transformedDF.loc[0, 'installment'] / transformedDF.loc[0, 'annual_inc']
    transformedDF.loc[0, 'loan_to_income_ratio'] = transformedDF.loc[0, 'funded_amnt'] / transformedDF.loc[0, 'annual_inc']
    
    # Floats (general)
    transformedDF.loc[0, 'int_rate'] = float(data['int_rate'])
    transformedDF.loc[0, 'dti'] = float(data['dti'])
    transformedDF.loc[0, 'delinq_2yrs'] = float(data['delinq_2yrs'])
    transformedDF.loc[0, 'inq_last_6mths'] = float(data['inq_last_6mths'])
    transformedDF.loc[0, 'open_acc'] = float(data['open_acc'])
    transformedDF.loc[0, 'pub_rec'] = float(data['pub_rec'])
    transformedDF.loc[0, 'revol_util'] = float(data['revol_util'])
    transformedDF.loc[0, 'total_acc'] = float(data['total_acc'])
    transformedDF.loc[0, 'collections_12_mths_ex_med'] = float(data['collections_12_mths_ex_med'])
    transformedDF.loc[0, 'acc_now_delinq'] = float(data['acc_now_delinq'])
    transformedDF.loc[0, 'tot_coll_amt'] = float(data['tot_coll_amnt'])
    transformedDF.loc[0, 'tot_cur_bal'] = float(data['tot_cur_bal'])
    transformedDF.loc[0, 'total_rev_hi_lim'] = float(data['total_rev_hi_lim'])


    # Transforming categorical variables
    categorical_columns = LabelEncodingModel.feature_names_in_
    categorical_values = pd.DataFrame(LabelEncodingModel.transform(transformedDF[categorical_columns]),columns = categorical_columns)
    transformedDF[categorical_columns] = categorical_values

    #print(transformedDF)
    return transformedDF

def defaultOrNot(data,encodingmodel, classificationmodel):
    try:
        data = transform(data,encodingmodel,classificationmodel)
    except:
        print('Data not parsable, likely due to invalid inputs (such as all nulls)')
        return [-1]
    
    #print(classificationmodel.predict(data))

    try:
        return [False] if classificationmodel.predict(data) == "Fully Paid" else [True,data]
    except:
        print('Either [Model] Or [Data] is not properly joining.')
        #print(f'Model: [ {classificationmodel} ] \n Data: \n \t {data}')
        return [-1]

def maximizeLoan(data, regress_model):
    loss = round(regress_model.predict(data)[0],2)

    suggestion = outputNewSuggestion(data,regress_model)
    if suggestion['mid'] < 0:
        meth1 = round(suggestion["high"], 2)
    elif suggestion['mid'] > data['funded_amnt'].iloc[0]:
        meth1 = round(suggestion["low"], 2)
    else:
        meth1 = round(suggestion["mid"], 2)

    meth2 = stepData(data,regress_model)

    return loss, meth1, meth2

def outputNewSuggestion(data,model):
  PrecalculatedAverageError = 2175.6484141912088
  anticipatedLoss = model.predict(data)[0]

  result = data['funded_amnt'].iloc[0] + anticipatedLoss

  return {'low': result - PrecalculatedAverageError, 'mid': result, 'high': result + PrecalculatedAverageError}

matrixSize = 9
def generateGrid(LoanAmount,IntRate):
  if LoanAmount < 200:
    quantityStepSize = 10
  elif LoanAmount < 1500: 
    quantityStepSize = 100
  elif LoanAmount < 25000:
    quantityStepSize = 1000
  else: 
     quantityStepSize = (LoanAmount)/10
  
  rateStepSize = 1.0
  
  matrix = [
      [LoanAmount - quantityStepSize , IntRate - rateStepSize], [LoanAmount , IntRate - rateStepSize], [LoanAmount + quantityStepSize , IntRate - rateStepSize],
      [LoanAmount - quantityStepSize , IntRate ], [LoanAmount , IntRate], [LoanAmount + quantityStepSize , IntRate],
      [LoanAmount - quantityStepSize , IntRate + rateStepSize], [LoanAmount , IntRate + rateStepSize], [LoanAmount + quantityStepSize , IntRate + rateStepSize],
  ]
  return quantityStepSize, pd.DataFrame(data = matrix, columns = ['funded_amnt', 'int_rate'])

# Returns what the step down is, as well as what % of the results are "Fully Paid"
def accuracy(data, model):
    while data.shape[0] < 9:
        data = pd.concat([data,data.sample(1)], ignore_index= True)

    step, generatedGrid = generateGrid(data['funded_amnt'][0],data['int_rate'][0])
    data[['funded_amnt','int_rate']] = generatedGrid

    #print(data)

    predictions = pd.DataFrame(model.predict(data),columns= ['value'])

    return step, (predictions.query('value == "Fully Paid"').count()/matrixSize).value

# Model Must be PreFitted
def stepData(data,model):
  acc = 0.0
  while acc < 0.6 and data['funded_amnt'].iloc[0] > 0.0:
    step, acc = accuracy(data, model)
    if acc > 0.6:
      break
    if data['funded_amnt'].iloc[0] < 0.0:
      break

    data['funded_amnt'] -= step
    data['installment'] = round((data['funded_amnt'] / data['term']) * (1.0 + data['int_rate']/(36*60/data['term'])), 2)
    data['loan_to_income_ratio'] = data['funded_amnt'] / data['annual_inc']
    data['installment_to_income_ratio'] = data['installment'] / data['annual_inc']

  return(data['funded_amnt'].iloc[0])


if __name__ == "__main__":
    print("This should not have been reached")

