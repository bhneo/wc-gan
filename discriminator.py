from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D, Lambda
from keras.layers import BatchNormalization, Add, Embedding, Concatenate, UpSampling2D, AveragePooling2D, Subtract

from gan.conditional_layers import cond_resblock, ConditionalConv11, ConditionalDense, ConditionalCenterScale,\
                                        get_separable_conv, ConditionalDepthwiseConv2D, DecorelationNormalization
from gan.spectral_normalized_layers import SNConv2D, SNDense, SNConditionalConv11, SNCondtionalDense, SNEmbeding, SNConditionalDepthwiseConv2D, SNConditionalConv2D, ConditionalConv2D
from gan.layer_utils import glorot_init, GlobalSumPooling2D
from functools import partial
import keras.backend as K


def make_discriminator(input_image_shape, input_cls_shape=(1, ), block_sizes=(128, 128, 128, 128),
                       resamples=('DOWN', "DOWN", "SAME", "SAME"), class_agnostic_blocks=4,
                       number_of_classes=10, type='AC_GAN', norm='n', after_norm='n', spectral=False,
                       conditional_bottleneck=False, unconditional_bottleneck=False,
                       conditional_shortcut=False, unconditional_shortcut=True,
                       fully_diff_spectral=False, spectral_iterations=1, conv_singular=True,
                       sum_pool=False, renorm_for_cond_singular=False, cls_branch=False, agnostic_stream=False):

    assert conditional_shortcut or unconditional_shortcut
    assert len(block_sizes) == len(resamples)
    x = Input(input_image_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if spectral:
        conv_layer = partial(SNConv2D, conv_singular=conv_singular,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        cond_dence_layer = partial(SNCondtionalDense,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations,
                              renormalize=renorm_for_cond_singular)
        cond_conv_layer = partial(SNConditionalConv11,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations,
                              renormalize=renorm_for_cond_singular)

        dence_layer = partial(SNDense,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        emb_layer = partial(SNEmbeding, fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        depthwise_layer = partial(SNConditionalDepthwiseConv2D,
                                  fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
    else:
        conv_layer = Conv2D
        cond_dence_layer = ConditionalDense
        cond_conv_layer = ConditionalConv11
        dence_layer = Dense
        emb_layer = Embedding
        depthwise_layer = ConditionalDepthwiseConv2D

    assert norm in ['n', 'cb', 'd', 'ub']
    assert after_norm in ['cs', 'conv', 'n']
    if norm == 'n':
        norm_layer = lambda axis, name: (lambda inp: inp)
    elif norm == 'cb':
        norm_layer = lambda axis, name: BatchNormalization(axis=axis, center=False, scale=False, name=name)
    elif norm == 'ub':
        norm_layer = BatchNormalization
    elif norm == 'd':
        norm_layer = lambda axis, name: DecorelationNormalization(name=name)

    if after_norm == 'cs':
        after_norm_layer =  lambda axis, name: lambda x: ConditionalCenterScale(number_of_classes=number_of_classes,
                                                                     axis=axis, name=name)([x, cls])
    elif after_norm == 'conv':
        after_norm_layer = lambda axis, name: lambda x: cond_conv_layer(number_of_classes=number_of_classes,
                                                                name=name, filters=K.int_shape(x)[axis])([x, cls])
    elif after_norm == 'n':
        after_norm_layer = lambda axis, name: lambda x: x

    def bn(axis, name):
        def stack(inp):
            out = inp
            out = norm_layer(axis=axis, name=name + '_npart')(out)
            out = after_norm_layer(axis=axis, name=name + '_repart')(out)
            return out
        return stack


    conv_layer_cls = partial(get_separable_conv, number_of_classes=number_of_classes, cls=cls,
                                                conv_layer=depthwise_layer, conv11_layer=cond_conv_layer,
                                                conditional_conv11=True, conditional_conv=True)
    y = x
    i = 0
    for block_size, resample in zip(block_sizes, resamples):
        input_dim = K.int_shape(y)[-1]
        uncond_shortcut = (input_dim != block_size) or (resample != "SAME")
        uncond_shortcut = unconditional_shortcut and uncond_shortcut

        y = cond_resblock(y, cls, kernel_size=(3, 3), resample=resample, nfilters=block_size,
                          number_of_classes=number_of_classes, name='Discriminator.' + str(i), norm=bn,
                          is_first=(i==0), conv_layer=conv_layer, cond_conv_layer=cond_conv_layer,
                          cond_bottleneck=conditional_bottleneck, uncond_bottleneck=unconditional_bottleneck,
                          cond_shortcut=conditional_shortcut, uncond_shortcut=uncond_shortcut,
                          cls_conv=conv_layer_cls if cls_branch else None)
        if i == class_agnostic_blocks - 1:
            y_c = y
        i += 1


    for block_size, resample in zip(block_sizes[class_agnostic_blocks:], resamples[class_agnostic_blocks:]):
        input_dim = K.int_shape(y)[-1]
        uncond_shortcut = (input_dim != block_size) or (resample != "SAME")

        y_c = cond_resblock(y_c, cls, kernel_size=(3, 3), resample=resample, nfilters=block_size,
                          number_of_classes=number_of_classes, name='CDiscriminator.' + str(i), norm=bn,
                          is_first=(i==0), conv_layer=conv_layer_cls, cond_conv_layer=None,
                          cond_bottleneck=False, uncond_bottleneck=False,
                          cond_shortcut=False, uncond_shortcut=uncond_shortcut)
        i += 1

    y = Activation('relu')(y)
    if sum_pool:
        y = GlobalSumPooling2D()(y)
    else: 
        y = GlobalAveragePooling2D()(y)

    if type == 'AC_GAN':
        cls_out = Dense(units=number_of_classes, use_bias=True, kernel_initializer=glorot_init)(y)
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=x, outputs=[out, cls_out])
    elif type == "PROJECTIVE":
        emb = emb_layer(input_dim = number_of_classes, output_dim = block_sizes[-1])(cls)
        phi = Lambda(lambda inp: K.sum(inp[1] * K.expand_dims(inp[0], axis=1), axis=2), output_shape=(1, ))([y, emb])
        psi = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        out = Add()([phi, psi])
        return Model(inputs=[x,cls], outputs=[out])
    elif type is None:
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=[x], outputs=[out])
    else:
        y_c = Activation('relu')(y_c)
        if sum_pool:
            y_c = GlobalSumPooling2D()(y_c)
        else:
            y_c = GlobalAveragePooling2D()(y_c)

        emb = emb_layer(input_dim = number_of_classes, output_dim = block_sizes[-1])(cls)

        out_phi = Lambda(lambda inp: K.sum(inp[1] * K.expand_dims(inp[0], axis=1), axis=2), output_shape=(1, ))([y_c, emb])
        out_psi = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        if agnostic_stream:
            out = Add()([out_phi, out_psi])
        else:
            out = out_phi
        return Model(inputs=[x,cls], outputs=[out])
