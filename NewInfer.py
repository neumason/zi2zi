# -*- coding: utf-8 -*-
DEFAULT_CHARSET = "./charset/txt9169.json" #字符集文件路径
class FromFont2Image(object):
    def __init__(self):
        import  json

        self.data =json.load(open(DEFAULT_CHARSET))["gbk"]
        print("examples -> %d" % len(self.data))
    def get_FontImage(self):
        import numpy as np
        from PIL import ImageFont
        src="font/mubiao.otf"#字体路径
        font = ImageFont.truetype(src, size=150)
        for c in self.data:
            from PIL import Image
            from PIL import ImageDraw
            canvas_size=256
            x_offset=20
            y_offset=20
            img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((x_offset, y_offset), c, (0, 0, 0), font=font)
            # example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
            # example_img.paste(img, (0, 0))
            # example_img.paste(img, (canvas_size, 0))
            img_A=np.array(img).astype(np.float)
            img_A=(img_A / 127.5) - 1.
            img=np.concatenate([img_A, img_A], axis=2)
            img=np.array([img]).astype(np.float32)

            yield c,img



    def newinfer(self, source_obj, embedding_ids, model_dir, save_dir):
        from .dataset import FromFont2Image
        image = FromFont2Image()

        tf.global_variables_initializer().run()
        saver = tf.train.Saver(var_list=self.retrieve_generator_vars())
        self.restore_model(saver, model_dir)

        def save_imgs(imgs, count):
            p = os.path.join(save_dir, "inferred_%04d.png" % count)
            save_concat_images(imgs, img_path=p)
            print("generated images saved at %s" % p)
        count = 0
        batch_buffer = list()
        for labels, source_imgs in image.get_FontImage():
            fake_imgs = self.generate_fake_samples(source_imgs, labels)[0] #生成
            merged_fake_images = merge(scale_back(fake_imgs), [self.batch_size, 1])
            batch_buffer.append(merged_fake_images)
            if len(batch_buffer) == 10:
                save_imgs(batch_buffer, count)
                batch_buffer = list()
            count += 1
        if batch_buffer:
            # last batch
            save_imgs(batch_buffer, count)
if __name__=="__main__":
    img=FromFont2Image()
    for i,j in img.get_FontImage():

        print(j.shape)
        print(type(j))