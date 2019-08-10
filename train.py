import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# 配置
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)
layers = tf.keras.layers
BATCH_SIZE = 256

# load training data
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5     # 规约到 -1~1
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)


# Generator model
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        dense1 = layers.Dense(7*7*256, use_bias=False, name="dense1")
        bn1 = layers.BatchNormalization(name="bn1")
        ac1 = layers.LeakyReLU(name="ac1")
        reshape1 = layers.Reshape(target_shape=(7, 7, 256), name="reshape1")

        dconv2 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, name="dconv2")
        bn2 = layers.BatchNormalization(name="bn2")
        ac2 = layers.LeakyReLU(name="ac2")

        dconv3 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, name="dconv3")
        bn3 = layers.BatchNormalization(name="bn3")
        ac3 = layers.LeakyReLU(name="ac3")
        # 使用 tanh 激活函数. 同时真实的图像也被规约到 -1~1 之间
        dconv4 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                        use_bias=False, activation='tanh', name="dconv4")   # -1 ~ 1

        self.model = tf.keras.Sequential([dense1, bn1, ac1, reshape1,
                                          dconv2, bn2, ac2,
                                          dconv3, bn3, ac3, dconv4])

    def call(self, x):
        x = self.model(x)
        return x


# Discriminator model
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        conv1 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', name='conv1')
        ac1 = layers.LeakyReLU(name='ac1')
        drop1 = layers.Dropout(0.3, name='drop1')

        conv2 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', name='conv2')
        ac2 = layers.LeakyReLU(name='ac2')
        drop2 = layers.Dropout(0.3, name='drop2')

        f1 = layers.Flatten(name='f1')
        dense = layers.Dense(1, name='dense')
        self.model = tf.keras.Sequential([conv1, ac1, drop1, conv2, ac2, drop2, f1, dense])
        self.model_test = tf.keras.Sequential([conv1, ac1, conv2, ac2, f1, dense])

    def call(self, x, training=True):
        if training:
            x = self.model(x)
        else:
            x = self.model_test(x)
        return x


def discriminator_loss(real_output, fake_output):
    """将real_output标记为1，fake_output标记为0，使用cross_entropy求损失"""
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(real_output), logits=real_output)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(fake_output), logits=fake_output)
    return tf.reduce_mean(real_loss) + tf.reduce_mean(fake_loss)


def generator_loss(fake_output):
    """generator希望自己的输出fake_output能够被标记为1"""
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_output), logits=fake_output)
    return tf.reduce_mean(g_loss)


# 初始化类, optimizer, checkpoint
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

checkpoint_dir = './training_checkpoint'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


# Training Loop
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16

# input to generator, keep fixed to test the quality
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def train_step(images):
    """迭代训练1个批量的数据"""
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    # Generator和Discriminator分别优化
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        print("generator loss:", gen_loss.numpy(), ", discriminator loss:", disc_loss.numpy())

    # compute gradient
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # apply gradient
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input):
    """调用model，即generator，以seed作为输入，产生16张图片存储"""
    predictions = model(test_input)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):    # 16张图片
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')   # 将 -1~1 变化为 0-255
        plt.axis("off")
    plt.savefig(os.path.join("generated_mnist", "image_at_epoch_{:04d}.png".format(epoch)))


def train():
    for epoch in range(EPOCHS):
        print("\nEpoch:"+str(epoch)+"  -----")
        # 在数据集上进行一轮迭代
        for image_batch in train_dataset:
            train_step(image_batch)
        # 生成图片
        generate_and_save_images(generator, epoch+1, seed)
        # save checkpoints
        if (epoch+1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
    train()

    # restore
    # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


