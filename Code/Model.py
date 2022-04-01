from keras.layers import Conv2D, AveragePooling2D, UpSampling2D, Concatenate, Average, Maximum
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras import regularizers

def siteFactorResidualBlock(x, f, s):
    x_residual = Conv2D(filters=f,kernel_size=(4,4), padding='same', strides=s, kernel_regularizer=regularizers.l2(1e-2))(x)
    x_residual = LeakyReLU(alpha=0.2)(x_residual)
    x_residual = Conv2D(filters=f, kernel_size=(4,4), padding='same', strides=1, kernel_regularizer=regularizers.l2(1e-2))(x_residual)
    x_shortcut = Conv2D(f, kernel_size=(1,1), strides=s)(x)
    x_residual = add([x_residual, x_shortcut])
    x_residual = LeakyReLU(alpha=0.2)(x_residual)
    return x_residual

def branFactorResidualBlock(x, f, s):
    x_shortcut = Conv2D(f, kernel_size=(1,1), strides=s)(x)
    x_residual = Conv2D(filters=f, 
                        kernel_size=(4,4), 
                        padding='same', 
                        strides=s, 
                        kernel_regularizer=regularizers.l2(1e-2))(x)
    x_residual = InstanceNormalization()(LeakyReLU(alpha=0.2)(x_residual))
    x_residual = Conv2D(filters=f, 
                        kernel_size=(4,4), 
                        padding='same', 
                        strides=1, 
                        kernel_regularizer=regularizers.l2(1e-2))(x_residual)
    x_residual = add([x_residual, x_shortcut])
    x_residual = InstanceNormalization()(LeakyReLU(alpha=0.2)(x_residual))
    return x_residual

def getSiteFactorMixer(inp1,inp2,inp3,filter):
    site_factor_mixture = UpSampling2D(size=(2, 2))(inp3)
    site_factor_mixture = concatenate([site_factor_mixture, inp2], axis=-1)
    site_factor_mixture = siteFactorResidualBlock(site_factor_mixture, filter*2, 1)  
    site_factor_mixture = UpSampling2D(size=(2, 2))(inp2)
    site_factor_mixture = concatenate([site_factor_mixture, inp1], axis=-1)
    site_factor_mixture = siteFactorResidualBlock(site_factor_mixture, filter*1, 1)  
    site_factor_mixture = UpSampling2D(size=(2,2))(site_factor_mixture)      
    site_factor_mixture = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(site_factor_mixture)
    site_factor_mixture = UpSampling2D(size=(2,2))(site_factor_mixture)
    site_factor_mixture = siteFactorResidualBlock(site_factor_mixture, filter*1, 1)
    return site_factor_mixture

def getBranFactorMixer(inp1,inp2,inp3,inp4,filter):
    bran_factor_mixture = UpSampling2D(size=(2, 2))(inp4)
    bran_factor_mixture = concatenate([bran_factor_mixture, inp3], axis=-1)
    bran_factor_mixture = branFactorResidualBlock(bran_factor_mixture, filter*4, 1)
    bran_factor_mixture = UpSampling2D(size=(2, 2))(bran_factor_mixture)
    bran_factor_mixture = concatenate([bran_factor_mixture, inp2], axis=-1)
    bran_factor_mixture = branFactorResidualBlock(bran_factor_mixture, filter*2, 1)  
    bran_factor_mixture = UpSampling2D(size=(2, 2))(bran_factor_mixture)
    bran_factor_mixture = concatenate([bran_factor_mixture, inp1], axis=-1)
    bran_factor_mixture = branFactorResidualBlock(bran_factor_mixture, filter*1, 1)      
    bran_factor_mixture = UpSampling2D(size=(2,2))(bran_factor_mixture)      
    bran_factor_mixture = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(bran_factor_mixture)
    bran_factor_mixture = branFactorResidualBlock(bran_factor_mixture, filter*1, 1)      
    return bran_factor_mixture

def getSiteFactorEncoder(image_shape, filter, site):
    input_data = Input(shape = image_shape)       
    feature_output1 = AveragePooling2D((2,2))(input_data)        
    feature_output2 = siteFactorResidualBlock(feature_output1, filter*1, 2)        
    feature_output3 = siteFactorResidualBlock(feature_output2, filter*2, 2)        
    feature_output4 = siteFactorResidualBlock(feature_output3, filter*4, 2)
    return Model(input_data, [feature_output2, feature_output3, feature_output4], name='{}_site_factor_encoder'.format(site))

def getBranFactorEncoder(image_shape, filter, site):
    input_data = Input(shape = image_shape)            
    feature_output1 = branFactorResidualBlock(input_data, filter*1, 2)        
    feature_output2 = branFactorResidualBlock(feature_output1, filter*2, 2)        
    feature_output3 = branFactorResidualBlock(feature_output2, filter*4, 2)     
    feature_output4 = branFactorResidualBlock(feature_output3, filter*8, 2)
    return Model(input_data, [feature_output1, feature_output2, feature_output3,feature_output4], name='{}_brain_factor_encoder'.format(site))

def getDecoder(image_shape, filter, site):
    l1 = image_shape[0]//2
    l2 = image_shape[1]//2
    # Site factor input group
    site_factor_input1 = Input(shape=(l1//2,l2//2, filter*1))
    site_factor_input2 = Input(shape=(l1//4,l2//4, filter*2))
    site_factor_input3 = Input(shape=(l1//8,l2//8, filter*4))
    # Brain factor input group
    bran_factor_input1 = Input(shape=(l1//1,l2//1, filter*1))
    bran_factor_input2 = Input(shape=(l1//2,l2//2, filter*2))
    bran_factor_input3 = Input(shape=(l1//4,l2//4, filter*4))
    bran_factor_input4 = Input(shape=(l1//8,l2//8, filter*8))
    # Get site-factor and brain-factor mixer
    site_factor_mixture = getSiteFactorMixer(site_factor_input1,site_factor_input2,site_factor_input3,filter)
    bran_factor_mixture = getBranFactorMixer(bran_factor_input1,bran_factor_input2,bran_factor_input3,bran_factor_input4,filter)
    mean_features = Average()([bran_factor_mixture, site_factor_mixture])
    max__features = Maximum()([bran_factor_mixture, site_factor_mixture])   
    con__features = Concatenate()([site_factor_mixture, bran_factor_mixture, mean_features, max__features])     
    synthetic_img = branFactorResidualBlock(con__features, filter//2, 1)     
    synthetic_img = Conv2D(filters=3, kernel_size=(1, 1),  activation = 'tanh',padding='same')(synthetic_img)
    return Model([site_factor_input1, site_factor_input2, site_factor_input3, 
                  bran_factor_input1, bran_factor_input2, bran_factor_input3, bran_factor_input4], 
                  synthetic_img, name='{}_decoder'.format(site))





