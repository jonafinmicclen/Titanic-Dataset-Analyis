import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticModel:
    
    def test(self,featureMatrix,outcomeVector,significanceLevel):
        
        if significanceLevel >= 1:
            raise ValueError(f'significance level {significanceLevel} must be between 0 and 1')
        
        predictions = self.predict(featureMatrix)
        self.test_results = predictions
        
        accuracy_count = 0
        insignificance_count = 0 
        
        for prediction, outcome in zip(predictions, outcomeVector):
            if prediction > 0.5+significanceLevel and outcome == 1 or prediction < 0.5-significanceLevel and outcome == 0:
                accuracy_count += 1
            if 0.5-significanceLevel<prediction<0.5+significanceLevel:
                insignificance_count +=1
                
        self.accuracy = accuracy_count/(len(predictions)-insignificance_count)
        self.frequency_of_non_null_prediction = 1 - insignificance_count/len(predictions)
        return f'Model testing complete,\n{round(self.frequency_of_non_null_prediction*100,2)}% of the inputs are useable to the {round(significanceLevel*200,2)}% significance level\nThe model predicted accurately {self.accuracy*100}% of the time.'
    
    def report_model_status(self):
        return f'{self.convergency_status}\nModel contains {self.number_of_features} features.\nCurrent weights for model are {self.weights}.\nAverage gradient dLdw is {self.average_gradients}.'
    
    def fit(self, featureMatrix, outputVector, learningRate, accuracyGoal, maxIterations):
        
        #Check that number of features attribute is not yet created
        if hasattr(self, 'number_of_features'):
            raise Exception('Fit method not available for pre-trained models')
        
        #Get number of features from width of featureMatrix
        self.number_of_features = np.shape(featureMatrix)[1]
        
        #Create vector of all weights with random starting values
        self.weights = np.matrix(np.random.rand(self.number_of_features))
        
        #Adam implementation
        m = np.zeros_like(self.weights)
        v = np.zeros_like(self.weights)
        beta1 = 0.9  # Exponential decay rates for moment estimates
        beta2 = 0.999
        epsilon = 1e-8  # Small constant to prevent division by zero
        
        iterations = 1
        
        while True:
            
            #Calculate partial derivatives
            try:
                dLdw = np.array((sigmoid(featureMatrix*self.weights.T) - outputVector).T * featureMatrix)
            except:
                return f'Error in gradient calculation, model did not converge on iteration {iterations}.'
            
            #Calculate new average gradient
            current_avg_gradient = np.mean(np.absolute(dLdw))
            
            #Test stop conditions
            if iterations >= maxIterations:
                self.convergency_status = f'Non converging after {iterations} iterations.'
                self.average_gradients = current_avg_gradient
                return self.convergency_status
            
            if current_avg_gradient < accuracyGoal:
                self.convergency_status = f'Converged after {iterations} iterations.'
                self.average_gradients = current_avg_gradient
                return self.convergency_status
            
            # Adam update rules
            m = beta1 * m + (1 - beta1) * dLdw
            v = beta2 * v + (1 - beta2) * (dLdw ** 2)
            m_hat = m / (1 - beta1 ** iterations)
            v_hat = v / (1 - beta2 ** iterations)

            self.weights -= learningRate * m_hat / (np.sqrt(v_hat) + epsilon)
            
            iterations += 1
    
    def predict(self,featureMatrix):
        return sigmoid(featureMatrix*self.weights.T)
