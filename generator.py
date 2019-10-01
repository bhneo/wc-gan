from tensorflow.python.keras.models import Input, Model
from tensorflow.python.keras.layers import Dense, Reshape, Activation, Conv2D, Conv2DTranspose
from tensorflow.python.keras.layers import BatchNormalization, Add, Embedding, Concatenate

import numpy as np
from tensorflow.python.keras import backend as K

from gan.layer_utils import glorot_init, resblock, dcblock
from gan.conditional_layers import ConditionalConv11, DecorelationNormalization, ConditionalCenterScale, CenterScale, FactorizedConv11
from gan.spectral_normalized_layers import SNConv2D, SNConditionalConv11, SNDense, SNEmbeding, SNFactorizedConv11
from functools import partial


def create_norm(norm, after_norm, decomposition='zca', cls=None, number_of_classes=None, filters_emb = 10,
                uncoditional_conv_layer=Conv2D, conditional_conv_layer=ConditionalConv11,
                factor_conv_layer=FactorizedConv11):
    assert norm in ['n', 'b', 'd', 'dr']
    assert after_norm in ['ucs', 'ccs', 'uccs', 'uconv', 'fconv', 'ufconv', 'cconv', 'ucconv', 'ccsuconv', 'n']

    if norm == 'n':
        norm_layer = lambda axis, name: (lambda inp: inp)
    elif norm == 'b':
        norm_layer = lambda axis, name: BatchNormalization(axis=axis, center=False, scale=False, name=name)
    elif norm == 'd':
        norm_layer = lambda axis, name: DecorelationNormalization(name=name, decomposition=decomposition)
    elif norm == 'dr':
        norm_layer = lambda axis, name: DecorelationNormalization(name=name, renorm=True)

    if after_norm == 'ccs':
        after_norm_layer = lambda axis, name: lambda x: ConditionalCenterScale(number_of_classes=number_of_classes,
                                                                               axis=axis, name=name)([x, cls])
    elif after_norm == 'ucs':
        after_norm_layer = lambda axis, name: lambda x: CenterScale(axis=axis, name=name)(x)
    elif after_norm == 'uccs':
        def after_norm_layer(axis, name):
            def f(x):
                c = ConditionalCenterScale(number_of_classes=number_of_classes, axis=axis, name=name + '_c')([x, cls])
                u = CenterScale(axis=axis, name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif after_norm == 'cconv':
        after_norm_layer = lambda axis, name: lambda x: conditional_conv_layer(filters=K.int_shape(x)[axis],
                                                                          number_of_classes=number_of_classes,
                                                                          name=name)([x, cls])
    elif after_norm == 'fconv':
        after_norm_layer = lambda axis, name: lambda x: factor_conv_layer(number_of_classes=number_of_classes,
                                                                         name=name + '_c', filters=K.int_shape(x)[axis],
                                                                         filters_emb=filters_emb, use_bias=False)([x, cls])
    elif after_norm == 'uconv':
        after_norm_layer = lambda axis, name: lambda x: uncoditional_conv_layer(kernel_size=(1, 1),
                                                               filters=K.int_shape(x)[axis], name=name)(x)
    elif after_norm == 'ucconv':
        def after_norm_layer(axis, name):
            def f(x):
                c = conditional_conv_layer(number_of_classes=number_of_classes, name=name + '_c',
                                      filters=K.int_shape(x)[axis])([x, cls])
                u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif after_norm == 'ccsuconv':
        def after_norm_layer(axis, name):
            def f(x):
                c = ConditionalCenterScale(number_of_classes=number_of_classes, axis=axis, name=name + '_c')([x, cls])
                u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif after_norm == 'ufconv':
        def after_norm_layer(axis, name):
            def f(x):
                c = factor_conv_layer(number_of_classes=number_of_classes, name=name + '_c',
                                     filters=K.int_shape(x)[axis], filters_emb=filters_emb,
                                     use_bias=False)([x, cls])
                u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif after_norm == 'n':
        after_norm_layer = lambda axis, name: lambda x: x

    def result_norm(axis, name):
        def stack(inp):
            out = inp
            out = norm_layer(axis=axis, name=name + '_npart')(out)
            out = after_norm_layer(axis=axis, name=name + '_repart')(out)
            return out
        return stack

    return result_norm


def make_generator(input_noise_shape=(128,), output_channels=3, input_cls_shape=(1, ),
                   block_sizes=(128, 128, 128), resamples=("UP", "UP", "UP"),
                   first_block_shape=(4, 4, 128), number_of_classes=10, concat_cls=False,
                   block_norm='u', block_after_norm='cs', filters_emb=10,
                   last_norm='u', last_after_norm='cs', decomposition='cholesky', gan_type=None, arch='res',
                   spectral=False, fully_diff_spectral=False, spectral_iterations=1, conv_singular=True,):

    assert arch in ['res', 'dcgan']
    inp = Input(input_noise_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if spectral:
        conv_layer = partial(SNConv2D, conv_singular=conv_singular,
                             fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        cond_conv_layer = partial(SNConditionalConv11,
                                  fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        dense_layer = partial(SNDense,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        emb_layer = partial(SNEmbeding, fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        factor_conv_layer = partial(SNFactorizedConv11,
                                    fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
    else:
        conv_layer = Conv2D
        cond_conv_layer = ConditionalConv11
        dense_layer = Dense
        emb_layer = Embedding
        factor_conv_layer = FactorizedConv11

    if concat_cls:
        y = emb_layer(input_dim=number_of_classes, output_dim=first_block_shape[-1])(cls)
        y = Reshape((first_block_shape[-1], ))(y)
        y = Concatenate(axis=-1)([y, inp])
    else:
        y = inp

    y = dense_layer(units=np.prod(first_block_shape), kernel_initializer=glorot_init)(y)
    y = Reshape(first_block_shape)(y)

    block_norm_layer = create_norm(block_norm, block_after_norm, decomposition=decomposition, cls=cls,
                                   number_of_classes=number_of_classes, filters_emb=filters_emb,
                                   uncoditional_conv_layer=conv_layer, conditional_conv_layer=cond_conv_layer,
                                   factor_conv_layer=factor_conv_layer)

    last_norm_layer = create_norm(last_norm, last_after_norm, decomposition=decomposition, cls=cls,
                                  number_of_classes=number_of_classes, filters_emb=filters_emb,
                                  uncoditional_conv_layer=conv_layer, conditional_conv_layer=cond_conv_layer,
                                  factor_conv_layer=factor_conv_layer)

    i = 0
    for block_size, resample in zip(block_sizes, resamples):
        if arch == 'res':
            y = resblock(y, kernel_size=(3, 3), resample=resample,
                         nfilters=block_size, name='Generator.' + str(i),
                         norm=block_norm_layer, is_first=False, conv_layer=conv_layer)
        else:
            # TODO: SN DECONV
            y = dcblock(y, kernel_size=(4, 4), resample=resample,
                        nfilters=block_size, name='Generator.' + str(i),
                        norm=block_norm_layer, is_first=False, conv_layer=Conv2DTranspose)
        i += 1

    y = last_norm_layer(axis=-1, name='Generator.BN.Final')(y)
    y = Activation('relu')(y)
    output = conv_layer(filters=output_channels, kernel_size=(3, 3), name='Generator.Final',
                        kernel_initializer=glorot_init, use_bias=True, padding='same')(y)
    output = Activation('tanh')(output)

    if gan_type is None:
        return Model(inputs=[inp], outputs=output)
    else:
        return Model(inputs=[inp, cls], outputs=output)

