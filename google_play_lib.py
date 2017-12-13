import gzip
import time
import random
import requests
from lxml import html as lxmlhtml
from bs4 import BeautifulSoup as bs

ualist_string = '''Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36
Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36
Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36
Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36
Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2226.0 Safari/537.36
Mozilla/5.0 (Windows NT 6.4; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36
Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36
Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2224.3 Safari/537.36'''
ualist = ualist_string.split('\n')
'''
    `category` VARCHAR(32) NOT NULL,
    `package_name` TEXT NOT NULL,
    `name` TEXT DEFAULT NULL,
    `author` TEXT DEFAULT NULL,
    `version` TEXT DEFAULT NULL,
    `rate` TEXT DEFAULT NULL,
    `size` TEXT DEFAULT NULL,
    `content_rating` INT UNSIGNED DEFAULT 0,
'''

google_play_info_base_url = "https://play.google.com/store/apps/details?id={package_name}&hl={hl}"
google_play_search_base_url = "https://play.google.com/store/search?q={q}&c=apps&hl={hl}&gl={gl}"


class SearchEntity:
    def __init__(self, pkg_name, hl):
        url = google_play_info_base_url.format(package_name=pkg_name, hl=hl)
        headers = {'User-Agent': ualist[random.randint(0, len(ualist) - 1)]}
        html = requests.get(url, headers=headers).text
        self.pkg_name = pkg_name
        self.hl = hl
        self.bs_obj = bs(html, 'lxml').select_one('div.details-wrapper.apps.square-cover')

    @property
    def package_name(self):
        return self.pkg_name

    @property
    def name(self):
        return self.bs_obj.select_one('div.id-app-title').text.strip()

    @property
    def language(self):
        return self.hl.replace('-', '_')

    @property
    def publisher(self):
        return self.bs_obj.select_one('a.document-subtitle.primary').text.strip()

    @property
    def is_game(self):
        return 'GAME' in self.bs_obj.select_one('a.document-subtitle.category').attrs['href']\
                                                                               .replace('_', '/').split('/')

    @property
    def icon(self):
        return "https:%s=w%s" % (self.bs_obj.select_one('img.cover-image').attrs['src'][:-8], 500)

    @property
    def images(self):
        return ["https:%s=h%s" % (i.attrs['src'][:-8], 1500) for i in self.bs_obj.select('img.screenshot')]

    @property
    def description(self):
        description = self.bs_obj.select_one('div.show-more-content.text-body')
        [i.replace_with('\n') for i in description.findAll('br')]
        return description.text.strip()

    # @returns((dict, None))
    def get(self):
        return {
            'pkg_name': self.package_name,
            'app_name': {
                'name': self.name,
                'lang': self.language
            },
            'publisher': self.publisher,
            'icon': self.icon,
            'images': self.images,
            'description': self.description
        } if self.is_game else None


class GooglePlaySearch:
    def __init__(self, keywords=None, hl='zh-TW', gl='tw'):
        self.hl = hl
        self.pkg_name = []
        html = ''

        if keywords is not None:
            for url in [google_play_search_base_url.format(q=i, hl=hl, gl=gl) for i in keywords]:
                headers = {'User-Agent': ualist[random.randint(0, len(ualist) - 1)]}
                html = requests.get(url, headers=headers).text
                self.pkg_name += [i.attrs['data-docid'] for i in bs(html, 'html.parser').select('div.card.apps')[:5]]
        else:
            data = dict(start=0, num=100, numChildren=0, cctcss="square-cover", cllayout="NORMAL", ipf=1, xhr=1)
            urls = [
                'https://play.google.com/store/apps/category/GAME/collection/topselling_free',
                'https://play.google.com/store/apps/category/GAME/collection/topselling_paid',
                'https://play.google.com/store/apps/category/GAME/collection/topgrossing'
            ]
            for url in urls:
                headers = {'User-Agent': ualist[random.randint(0, len(ualist) - 1)]}
                html = requests.post(url, data=data, headers=headers).text
            self.pkg_name += [i.attrs['data-docid'] for i in bs(html, 'html.parser').select('div.card.apps')]

        self.pkg_name = sorted(set(self.pkg_name))
        print("Total: %s packages..." % self.pkg_name.__len__())

    def __iter__(self):
        for e in self.pkg_name:
            entity = SearchEntity(pkg_name=e, hl=self.hl)
            rs = entity.get()
            if not rs:
                continue
            yield rs




    # for row in GooglePlaySearch():
    #     folder_name = row['pkg_name'].replace('.', '-')
    #     if folder_name not in os.listdir('dataset/training_pics'):
    #         os.mkdir('dataset/training_pics/' + folder_name)
    #         for nb, img in enumerate(row['images']):
    #             print("Gathering %s_%s.jpg" % (row['app_name']['name'], nb))
    #             img_write('dataset/training_pics/' + folder_name + "/%s_%s.jpg" % (row['app_name']['name'], nb))

