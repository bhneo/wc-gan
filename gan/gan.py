import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model


class GAN(object):
    def __init__(self, generator, discriminator,
                 generator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0, beta_2=0.9),
                 discriminator_optimizer=tf.keras.optimizers.Adam(2e-4, beta_1=0, beta_2=0.9),
                 generator_adversarial_objective='ns-gan',
                 discriminator_adversarial_objective='ns-gan',
                 gradient_penalty_weight=10,
                 gradient_penalty_type='dragan',
                 # additional_inputs_for_generator_train=[],
                 # additional_inputs_for_discriminator_train=[],
                 custom_objects={},
                 lr_decay_schedule_generator=lambda iter: 1.0,
                 lr_decay_schedule_discriminator=lambda iter: 1.0,
                 **kwargs):
        assert generator_adversarial_objective in ['ns-gan', 'lsgan', 'wgan', 'hinge']
        assert discriminator_adversarial_objective in ['ns-gan', 'lsgan', 'wgan', 'hinge']
        assert gradient_penalty_type in ['dragan', 'wgan-gp']

        if type(generator) == str:
            self.generator = load_model(generator, custom_objects=custom_objects)
        else:
            self.generator = generator

        if type(discriminator) == str:
            self.discriminator = load_model(discriminator, custom_objects=custom_objects)
        else:
            self.discriminator = discriminator

        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        generator_input = self.generator.input
        discriminator_input = self.discriminator.input

        if type(generator_input) == list:
            self.generator_input = generator_input
        else:
            self.generator_input = [generator_input]

        if type(discriminator_input) == list:
            self.discriminator_input = discriminator_input
        else:
            self.discriminator_input = [discriminator_input]

        self.generator_metric_names = []
        self.discriminator_metric_names = []

        self.generator_adversarial_loss_func = self.get_generator_adversarial_loss(generator_adversarial_objective)
        self.discriminator_adversarial_loss_func = self.get_discriminator_adversarial_loss(discriminator_adversarial_objective)

        # self.additional_inputs_for_generator_train = additional_inputs_for_generator_train
        # self.additional_inputs_for_discriminator_train = additional_inputs_for_discriminator_train
        self.gradient_penalty_weight = gradient_penalty_weight
        self.gradient_penalty_type = gradient_penalty_type

        self.lr_decay_schedule_generator = lr_decay_schedule_generator
        self.lr_decay_schedule_discriminator = lr_decay_schedule_discriminator

    def get_generator_adversarial_loss(self, loss_type):
        self.generator_metric_names.append('fake')
        if loss_type == 'ns-gan':
            def ns_loss(logits):
                labels = tf.ones_like(logits)
                return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
            return ns_loss
        if loss_type == 'lsgan':
            def ls_loss(logits):
                return tf.reduce_mean((logits - 1) ** 2)
            return ls_loss
        if loss_type == 'wgan':
            def wgan(logits):
                return -tf.reduce_mean(logits)
            return wgan
        if loss_type == 'hinge':
            def hinge(logits):
                return -tf.reduce_mean(logits)
            return hinge
        else:
            return None

    def get_discriminator_adversarial_loss(self, loss_type):
        self.discriminator_metric_names.append('true')
        self.discriminator_metric_names.append('fake')
        if loss_type == 'ns-gan':
            def ns_loss_true(logits):
                labels = tf.ones_like(logits)
                return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

            def ns_loss_fake(logits):
                labels = tf.zeros_like(logits)
                return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
            return [ns_loss_true, ns_loss_fake]
        if loss_type == 'lsgan':
            def ls_loss_true(logits):
                return tf.reduce_mean((logits - 1) ** 2)

            def ls_loss_fake(logits):
                return tf.reduce_mean(logits ** 2)
            return [ls_loss_true, ls_loss_fake]
        if loss_type == 'wgan':
            def wgan_loss_true(logits):
                return -tf.reduce_mean(logits)

            def wgan_loss_fake(logits):
                return tf.reduce_mean(logits)
            return [wgan_loss_true, wgan_loss_fake]
        if loss_type == 'hinge':
            def hinge_loss_true(logits):
                return tf.reduce_mean(tf.maximum(0.0, 1.0 - logits))

            def hinge_loss_fake(logits):
                return tf.reduce_mean(tf.maximum(0.0, 1.0 + logits))
            return [hinge_loss_true, hinge_loss_fake]
        else:
            return None

    @tf.function
    def get_gradient_penalty_loss(self, discriminator_input, generator_output):
        if self.gradient_penalty_weight == 0:
            return []

        if type(discriminator_input) == list:
            batch_size = tf.shape(discriminator_input[0])[0]
            ranks = [len(inp.get_shape().as_list()) for inp in discriminator_input]
        else:
            batch_size = tf.shape(discriminator_input)[0]
            ranks = [len(discriminator_input.get_shape().as_list())]

        def cast_all(values, reference_type_vals):
            return [tf.cast(alpha, dtype=ref.dtype) for alpha, ref in zip(values, reference_type_vals)]

        def std_if_not_int(val):
            if val.dtype.is_integer:
                return 0
            else:
                return tf.stop_gradient(K.std(val, keepdims=True))

        def point_for_gp_wgan():
            weights = tf.random.uniform((batch_size, 1), minval=0, maxval=1)
            weights = [tf.reshape(weights, (-1, ) + (1, ) * (rank - 1)) for rank in ranks]
            weights = cast_all(weights, discriminator_input)
            points = [(w * r) + ((1 - w) * f) for r, f, w in zip(discriminator_input, generator_output, weights)]
            return points

        def points_for_dragan():
            alphas = tf.random.uniform((batch_size, 1), minval=0, maxval=1)
            alphas = [tf.reshape(alphas, (-1, ) + (1, ) * (rank - 1)) for rank in ranks]
            alphas = cast_all(alphas, discriminator_input)
            fake = [tf.random.uniform(tf.shape(t), minval=0, maxval=1) * std_if_not_int(t) * 0.5
                    for t in discriminator_input]
            fake = cast_all(fake, discriminator_input)
            points = [(w * r) + ((1 - w) * f) for r, f, w in zip(discriminator_input, fake, alphas)]
            return points

        points = {'wgan-gp': point_for_gp_wgan(), 'dragan': points_for_dragan()}
        points = points[self.gradient_penalty_type]

        gp_list = []
        with tf.GradientTape() as tape:
            disc_out = self.discriminator(points)
        if type(disc_out) != list:
            disc_out = [disc_out]
        gradients = tape.gradient(disc_out[0], points)

        for gradient in gradients:
            if gradient is None:
                continue
            gradient = tf.reshape(gradient, (batch_size, -1))
            gradient_l2_norm = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
            gradient_penalty = self.gradient_penalty_weight * tf.square(1 - gradient_l2_norm)
            gp_list.append(tf.reduce_mean(gradient_penalty))

        for i in range(len(gp_list)):
            self.discriminator_metric_names.append('gp_loss_' + str(i))
        return gp_list

    # def compile_intermediate_variables(self):
    #     self.generator_output = self.generator(self.generator_input)
    #     self.discriminator_fake_output = self.discriminator(self.generator_output)
    #     self.discriminator_real_output = self.discriminator(self.discriminator_input)

    # def intermediate_variables_to_lists(self):
    #     if type(self.generator_output) != list:
    #         self.generator_output = [self.generator_output]
    #     if type(self.discriminator_fake_output) != list:
    #         self.discriminator_fake_output = [self.discriminator_fake_output]
    #     if type(self.discriminator_real_output) != list:
    #         self.discriminator_real_output = [self.discriminator_real_output]

    def additional_generator_losses(self):
        return []

    def additional_discriminator_losses(self):
        return []

    # def collect_updates(self, model):
    #     updates = []
    #     for l in model.layers:
    #         updates += l.updates
    #     return updates

    def compile_generator_train_op(self):
        def update_lr():
            lr_update = (self.lr_decay_schedule_generator(self.generator_optimizer.iterations) *
                         K.get_value(self.generator_optimizer.lr))
            K.set_value(self.generator_optimizer.lr, lr_update)

        @tf.function
        def generator_train_op(generator_input):  # generator_input + additional_inputs_for_generator_train + train/test]
            loss_list = []
            with tf.GradientTape() as tape:
                generator_output = self.generator(generator_input + [True])
                discriminator_fake_output = self.discriminator([generator_output] + [True])
                if type(discriminator_fake_output) == list:
                    discriminator_fake_output = discriminator_fake_output[0]
                adversarial_loss = self.generator_adversarial_loss_func(discriminator_fake_output)
                loss_list.append(adversarial_loss)
                loss_list += self.additional_generator_losses()
            gradients = tape.gradient(loss_list, self.generator.trainable_weights)
            self.generator_optimizer.apply_gradients(zip(gradients, self.generator.trainable_weights))

            self.generator_loss_list = loss_list

            tf.py_function(update_lr, inp=[], Tout=[])
            return [sum(loss_list)] + loss_list
        return generator_train_op

    def compile_discriminator_train_op(self):
        def update_lr():
            lr_update = self.lr_decay_schedule_discriminator(self.discriminator_optimizer.iterations) * \
                        K.get_value(self.discriminator_optimizer.lr)
            K.set_value(self.discriminator_optimizer.lr, lr_update)

        @tf.function
        def discriminator_train_op(discriminator_input, generator_input):
            loss_list = []
            with tf.GradientTape() as tape:
                generator_output = self.generator(generator_input + [True])
                discriminator_fake_output = self.discriminator([generator_output, True])
                if type(discriminator_fake_output) == list:
                    discriminator_fake_output = discriminator_fake_output[0]
                adversarial_loss_fake = self.discriminator_adversarial_loss_func[1](discriminator_fake_output)

                discriminator_real_output = self.discriminator(discriminator_input + [True])
                if type(discriminator_real_output) == list:
                    discriminator_real_output = discriminator_real_output[0]
                adversarial_loss_real = self.discriminator_adversarial_loss_func[0](discriminator_real_output)

                loss_list += [adversarial_loss_real, adversarial_loss_fake]
                if self.gradient_penalty_weight != 0:
                    loss_list += self.get_gradient_penalty_loss(discriminator_input, generator_output)
                loss_list += self.additional_discriminator_losses()
            gradients = tape.gradient(loss_list, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_weights))

            tf.py_function(update_lr, inp=[], Tout=[])
            return [sum(loss_list)] + loss_list
        return discriminator_train_op

    def compile_generate_op(self):
        @tf.function
        def generate_op(generator_input, phase):
            generator_output = self.generator(generator_input + [phase])
            return generator_output
        return generate_op

    def compile_validate_op(self):
        @tf.function
        def validate_op(generator_input):
            loss_list = []
            generator_output = self.generator(generator_input + [True])
            discriminator_fake_output = self.discriminator([generator_output] + [True])
            if type(discriminator_fake_output) == list:
                discriminator_fake_output = discriminator_fake_output[0]
            adversarial_loss = self.generator_adversarial_loss_func(discriminator_fake_output)
            loss_list.append(adversarial_loss)
            loss_list += self.additional_generator_losses()
            return [sum(loss_list)] + loss_list
        return validate_op

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def get_losses_as_string(self, generator_losses, discriminator_losses):
        def combine(name_list, losses):
            losses = np.array(losses)
            if len(losses.shape) == 0:
                losses = losses.reshape((1, ))
            return '; '.join([name + ' = ' + str(loss) for name, loss in zip(name_list, losses)])
        generator_loss_str = combine(['Generator loss'] + self.generator_metric_names, generator_losses)
        discriminator_loss_str = combine(['Disciminator loss'] + self.discriminator_metric_names, discriminator_losses)
        return generator_loss_str, discriminator_loss_str
