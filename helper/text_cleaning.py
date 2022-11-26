import re


class Text_Cleaner:
    def __init__(self):
        self.text = None

    def preprocessor(self, text):
        self.text = text
        self.text = re.sub('<[^>]*>', '', self.text)
        emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', self.text)
        self.text = (re.sub('[\W]+', ' ', self.text.lower())
                     + ' '.join(emoticons).replace('-', ''))
        return self.text
