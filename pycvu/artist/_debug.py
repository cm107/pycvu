from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Type
import cv2
import os
from shutil import rmtree
import numpy as np

from pycvu.util import CvUtil, PilUtil, \
    LoadableImageMask, LoadableImageMaskHandler
from ..vector import Vector
from ..color import Color, HSV
from ..interval import Interval
from ..text_generator import TextGenerator
if TYPE_CHECKING:
    from ._artist import Artist

@classmethod
def debug(cls: Type[Artist]):
    cls.maskSetting.track = False; cls.maskSetting.canBeOccluded = True; cls.maskSetting.canOcclude = True

    imgHandlerPath = "imgHandler.json"
    if not os.path.isfile(imgHandlerPath):
        imgHandler = LoadableImageMaskHandler.from_wildcard(
            "symbol/*.png",
            Interval[HSV](HSV(0,0,0), HSV(359.9, 1, 0.9))
        )
        imgHandler.load_data()
        imgHandler.save(imgHandlerPath, includeData=True)
    else:
        imgHandler = LoadableImageMaskHandler.load(imgHandlerPath)
    imgHandlerRef = cls.context.register_variable(imgHandler)

    colorInterval: Interval[Color] = Interval[Color](Color.black, Color.white)

    img = np.zeros((500, 500, 3), dtype=np.uint8)
    cls.color = colorInterval.random()
    cls.thickness = 1

    p0 = Vector(200, 200); p1 = Vector(300, 200)
    r = int(Vector.Distance(p0, p1))
    center = 0.5 * (p0 + p1)
    width = 3 * r; height = 2 * r
    rectShape = Vector(width, height)
    c0 = center - 0.5 * rectShape
    c1 = center + 0.5 * rectShape

    drawer = cls(img)
    (
        drawer
        .circle(center=p0, radius=r)
        .circle(center=p1, radius=r)
        .line(p0, p1)
        .rectangle(c0, c1)
    )
    cls.color = colorInterval.random()
    cls.thickness = 4
    offset = (Vector.down + Vector.right).normalized * 50
    (
        drawer
        .circle(center=p0 + offset, radius=r, fill=True)
        .circle(center=p1 + offset, radius=r, fill=True)
        .line(p0 + offset, p1 + offset)
        .rectangle(c0 + offset, c1 + offset)
    )
    cls.color = colorInterval.random()
    offset = (Vector.down + Vector.left).normalized * 50
    (
        drawer
        .circle(center=p0 + offset, radius=r)
        .circle(center=p1 + offset, radius=r)
        .line(p0 + offset, p1 + offset)
        .rectangle(c0 + offset, c1 + offset, fill=True)
    )
    cls.color = colorInterval.random()
    offset = Vector.up * 100
    drawer.ellipse(
        center=center+offset, axis=(70, 30),
        angle=30, startAngle=90, endAngle=270,
        fill=True
    )
    drawer.resize(fx=2, fy=0.9)
    drawer.affine_rotate(45, adjustBorder=True)
    cls.color = colorInterval.random()
    cls.fontScale = 2.0
    cls.maskSetting.track = True
    drawer.text("Hello World!", org=(100, 100), rotation=0)
    drawer.text("Hello World!", org=(100, 100), rotation=Interval[float](-180, 180))
    drawer.text("Hello World!", org=(100, 200), bottomLeftOrigin=True)
    cls.maskSetting.track = False

    cls.maskSetting.track = True
    drawer.pil.text(text="荒唐無稽", position=(300, 300), rotation=Interval[float](0, 359))
    cls.maskSetting.track = False

    cls.color = Color(255, 0, 0)
    cls.PIL.fontSize = 50
    cls.PIL.hankoOutlineWidthRatio = 0.1
    cls.PIL.hankoMarginRatio = 0.1
    cls.maskSetting.track = True
    drawer.pil.hanko(text="合格", position=(300, 300+200))
    cls.maskSetting.track = False

    cls.color = Interval[HSV](HSV(0, 0.9375, 0.5), HSV(359.9, 1.0, 1.0))
    positionCallback = CvUtil.Callback.get_position_interval
    
    textGen = TextGenerator(
        characters="亜哀挨愛曖悪握圧扱宛嵐安案暗以衣位囲医依委威為畏胃尉異移萎偉椅彙意違維慰遺緯域育一壱逸茨芋引印因咽姻員院淫陰飲隠韻右宇羽雨唄鬱畝浦運雲永泳英映栄営詠影鋭衛易疫益液駅悦越謁閲円延沿炎怨宴媛援園煙猿遠鉛塩演縁艶汚王凹央応往押旺欧殴桜翁奥横岡屋億憶臆虞乙俺卸音恩温穏下化火加可仮何花佳価果河苛科架夏家荷華菓貨渦過嫁暇禍靴寡歌箇稼課蚊牙瓦我画芽賀雅餓介回灰会快戒改怪拐悔海界皆械絵開階塊楷解潰壊懐諧貝外劾害崖涯街慨蓋該概骸垣柿各角拡革格核殻郭覚較隔閣確獲嚇穫学岳楽額顎掛潟括活喝渇割葛滑褐轄且株釜鎌刈干刊甘汗缶完肝官冠巻看陥乾勘患貫寒喚堪換敢棺款間閑勧寛幹感漢慣管関歓監緩憾還館環簡観韓艦鑑丸含岸岩玩眼頑顔願企伎危机気岐希忌汽奇祈季紀軌既記起飢鬼帰基寄規亀喜幾揮期棋貴棄毀旗器畿輝機騎技宜偽欺義疑儀戯擬犠議菊吉喫詰却客脚逆虐九久及弓丘旧休吸朽臼求究泣急級糾宮救球給嗅窮牛去巨居拒拠挙虚許距魚御漁凶共叫狂京享供協況峡挟狭恐恭胸脅強教郷境橋矯鏡競響驚仰暁業凝曲局極玉巾斤均近金菌勤琴筋僅禁緊錦謹襟吟銀区句苦駆具惧愚空偶遇隅串屈掘窟熊繰君訓勲薫軍郡群兄刑形系径茎係型契計恵啓掲渓経蛍敬景軽傾携継詣慶憬稽憩警鶏芸迎鯨隙劇撃激桁欠穴血決結傑潔月犬件見券肩建研県倹兼剣拳軒健険圏堅検嫌献絹遣権憲賢謙鍵繭顕験懸元幻玄言弦限原現舷減源厳己戸古呼固股虎孤弧故枯個庫湖雇誇鼓錮顧五互午呉後娯悟碁語誤護口工公勾孔功巧広甲交光向后好江考行坑孝抗攻更効幸拘肯侯厚恒洪皇紅荒郊香候校耕航貢降高康控梗黄喉慌港硬絞項溝鉱構綱酵稿興衡鋼講購乞号合拷剛傲豪克告谷刻国黒穀酷獄骨駒込頃今困昆恨根婚混痕紺魂墾懇左佐沙査砂唆差詐鎖座挫才再災妻采砕宰栽彩採済祭斎細菜最裁債催塞歳載際埼在材剤財罪崎作削昨柵索策酢搾錯咲冊札刷刹拶殺察撮擦雑皿三山参桟蚕惨産傘散算酸賛残斬暫士子支止氏仕史司四市矢旨死糸至伺志私使刺始姉枝祉肢姿思指施師恣紙脂視紫詞歯嗣試詩資飼誌雌摯賜諮示字寺次耳自似児事侍治持時滋慈辞磁餌璽鹿式識軸七𠮟失室疾執湿嫉漆質実芝写社車舎者射捨赦斜煮遮謝邪蛇尺借酌釈爵若弱寂手主守朱取狩首殊珠酒腫種趣寿受呪授需儒樹収囚州舟秀周宗拾秋臭修袖終羞習週就衆集愁酬醜蹴襲十汁充住柔重従渋銃獣縦叔祝宿淑粛縮塾熟出述術俊春瞬旬巡盾准殉純循順準潤遵処初所書庶暑署緒諸女如助序叙徐除小升少召匠床抄肖尚招承昇松沼昭宵将消症祥称笑唱商渉章紹訟勝掌晶焼焦硝粧詔証象傷奨照詳彰障憧衝賞償礁鐘上丈冗条状乗城浄剰常情場畳蒸縄壌嬢錠譲醸色拭食植殖飾触嘱織職辱尻心申伸臣芯身辛侵信津神唇娠振浸真針深紳進森診寝慎新審震薪親人刃仁尽迅甚陣尋腎須図水吹垂炊帥粋衰推酔遂睡穂16随髄枢崇数据杉裾寸瀬是井世正生成西声制姓征性青斉政星牲省凄逝清盛婿晴勢聖誠精製誓静請整醒税夕斥石赤昔析席脊隻惜戚責跡積績籍切折拙窃接設雪摂節説舌絶千川仙占先宣専泉浅洗染扇栓旋船戦煎羨腺詮践箋銭潜線遷選薦繊鮮全前善然禅漸膳繕狙阻祖租素措粗組疎訴塑遡礎双壮早争走奏相荘草送倉捜挿桑巣掃曹曽爽窓創喪痩葬装僧想層総遭槽踪操燥霜騒藻造像増憎蔵贈臓即束足促則息捉速側測俗族属賊続卒率存村孫尊損他多汰打妥唾堕惰駄太対体耐待怠胎退帯泰堆袋逮替貸隊滞態戴大代台第題滝宅択沢卓拓託濯諾濁但達脱奪棚誰丹旦担単炭胆探淡短嘆端綻誕鍛団男段断弾暖談壇地池知値恥致遅痴稚置緻竹畜逐蓄築秩窒茶着嫡中仲虫沖宙忠抽注昼柱衷酎鋳駐著貯丁弔庁兆町長挑帳張彫眺釣頂鳥朝12貼超腸跳徴嘲潮澄調聴懲直勅捗沈珍朕陳賃鎮追椎墜通痛塚漬坪爪鶴低呈廷弟定底抵邸亭貞帝訂庭逓停偵堤提程艇締諦泥的笛摘滴適敵溺迭哲鉄徹撤天典店点展添転塡田伝殿電斗吐妬徒途都渡塗賭土奴努度怒刀冬灯当投豆東到逃倒凍唐島桃討透党悼盗陶塔搭棟湯痘登答等筒統稲踏糖頭謄藤闘騰同洞胴動堂童道働銅導瞳峠匿特得督徳篤毒独読栃凸突届屯豚頓貪鈍曇丼那奈内梨謎鍋南軟難二尼弐匂肉虹日入乳尿任妊忍認寧熱年念捻粘燃悩納能脳農濃把波派破覇馬婆罵拝杯背肺俳配排敗廃輩売倍梅培陪媒買賠白伯拍泊迫剝舶博薄麦漠縛爆箱箸畑肌八鉢発髪伐抜罰閥反半氾犯帆汎伴判坂阪板版班畔般販斑飯搬煩頒範繁藩晩番蛮盤比皮妃否批彼披肥非卑飛疲秘被悲扉費碑罷避尾眉美備微鼻膝肘匹必泌筆姫百氷表俵票評漂標苗秒病描猫品浜貧賓頻敏瓶不夫父付布扶府怖阜附訃負赴浮婦符富普腐敷膚賦譜侮武部舞封風伏服副幅復福腹複覆払沸仏物粉紛雰噴墳憤奮分文聞丙平兵併並柄陛閉塀幣弊蔽餅米壁璧癖別蔑片辺返変偏遍編便勉歩保哺捕補舗母募墓慕暮簿方包芳邦奉宝抱放法泡胞俸倣峰砲崩訪報蜂豊飽褒縫亡乏忙坊妨忘防房肪某冒剖紡望傍帽棒貿貌暴膨謀頰北木朴牧睦僕墨撲没勃堀本奔翻凡盆麻摩磨魔毎妹枚昧埋幕膜枕又末抹万満慢漫未味魅岬密蜜脈妙民眠矛務無夢霧娘名命明迷冥盟銘鳴滅免面綿麺茂模毛妄盲耗猛網目黙門紋問4冶夜野弥厄役約訳薬躍闇由油喩愉諭輸癒唯友有勇幽悠郵湧猶裕遊雄誘憂融優与予余誉預幼用羊妖洋要容庸揚揺葉陽溶腰様瘍踊窯養擁謡曜抑沃浴欲翌翼拉裸羅来雷頼絡落酪辣乱卵覧濫藍欄吏利里理痢裏履璃離陸立律慄略柳流留竜粒隆硫侶旅虜慮了両良料涼猟陵量僚領寮療瞭糧力緑林厘倫輪隣臨瑠涙累塁類令礼冷励戻例鈴零霊隷齢麗暦歴列劣烈裂恋連廉練錬呂炉賂路露老労弄郎朗浪廊楼漏籠六録麓論和話賄脇惑枠湾腕",
        textLength=2, allowRepetition=False
    )

    drawer.circle(
        center=positionCallback,
        radius=Interval[int](5, 100),
        fill=True,
        repeat=10
    )

    # for i in range(10):
    #     drawer.line(pt1=positionCallback, pt2=positionCallback)
    #     drawer.rectangle(pt1=positionCallback, pt2=positionCallback, fill=False)
    #     drawer.ellipse(
    #         center=positionCallback,
    #         axis=Interval[Vector[float]](Vector[float](5, 5), Vector[float](100, 100)),
    #         angle=Interval[float](0, 180),
    #         startAngle=Interval[float](0, 360),
    #         endAngle=Interval[float](0, 360),
    #         fill=False
    #     )
    #     drawer.pil.hanko(text=textGen, position=PilUtil.Callback.get_position_interval)

    cls.maskSetting.track = True
    drawer.overlay_image(
        imgHandlerRef, position=positionCallback,
        rotation=Interval[float](-180, 180),
        scale=Interval[float](0.5, 1.5),
        # noise=Interval[HSV](HSV(0,0,-50/255), HSV(0,0,50/255)),
        noise=Interval[int](-50, 50),
        repeat=4
    )
    cls.maskSetting.track = False

    drawer.line(pt1=positionCallback, pt2=positionCallback, repeat=10)
    cls.maskSetting.track = True
    drawer.rectangle(
        pt1=positionCallback, pt2=positionCallback, fill=False,
        rotation=Interval[float](-180, 180),
        repeat=10
    )
    cls.maskSetting.track = False
    drawer.ellipse(
        center=positionCallback,
        axis=Interval[Vector[float]](Vector[float](5, 5), Vector[float](100, 100)),
        angle=Interval[float](0, 180),
        startAngle=Interval[float](0, 360),
        endAngle=Interval[float](0, 360),
        fill=False, repeat=10
    )
    cls.PIL.fontSize = Interval[int](5, 40)
    cls.PIL.hankoOutlineWidthRatio = Interval[float](0.05, 0.2)
    cls.PIL.hankoMarginRatio = Interval[float](0.1, 0.5)
    cls.maskSetting.track = True
    drawer.pil.hanko(
        text=textGen, position=PilUtil.Callback.get_position_interval,
        rotation=Interval[float](0, 360),
        repeat=10
    )
    cls.maskSetting.track = False

    drawer.save('/tmp/artistDebugSave.json', saveImg=False, saveMeta=True)
    del drawer
    drawer = cls.load('/tmp/artistDebugSave.json', img=img, loadMeta=True) # Make sure save and load works.

    result, maskHandler = drawer.draw_and_get_masks()

    from ..polygon import Segmentation

    previewDump = 'previewDump'
    if os.path.isdir(previewDump):
        rmtree(previewDump)
    os.makedirs(previewDump, exist_ok=True)
    cv2.imwrite(f"{previewDump}/result.png", result)
    cv2.imwrite(f"{previewDump}/maskPreview.png", maskHandler.preview)
    for i, mask in enumerate(maskHandler):
        if mask._mask.sum() == 0:
            continue
        maskImg = mask.get_preview(showBBox=True, showContours=True, minNumPoints=6)
        numStr = str(i)
        while len(numStr) < 2:
            numStr = f"0{numStr}"
        cv2.imwrite(f"{previewDump}/mask{numStr}.png", maskImg)

    cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('result', 500, 500)
    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

@classmethod
def debug_loop(cls: Type[Artist], n: int=1):
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    drawer = cls.load('/tmp/artistDebugSave.json', img=img.copy(), loadMeta=True)

    previewLoopDump = 'previewLoopDump'
    if os.path.isdir(previewLoopDump):
        rmtree(previewLoopDump)
    os.makedirs(previewLoopDump, exist_ok=True)

    for k in range(n):
        result, maskHandler = drawer.draw_and_get_masks()

        previewDump = f'{previewLoopDump}/previewDump{k}'
        if os.path.isdir(previewDump):
            rmtree(previewDump)
        os.makedirs(previewDump, exist_ok=True)
        cv2.imwrite(f"{previewDump}/result.png", result)
        cv2.imwrite(f"{previewDump}/maskPreview.png", maskHandler.preview)
        for i, mask in enumerate(maskHandler):
            if mask._mask.sum() == 0:
                continue
            maskImg = mask.get_preview(showBBox=True, showContours=True, minNumPoints=6)
            numStr = str(i)
            while len(numStr) < 2:
                numStr = f"0{numStr}"
            cv2.imwrite(f"{previewDump}/mask{numStr}.png", maskImg)
