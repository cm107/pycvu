from __future__ import annotations
import random
from .base import Base, BaseHandler
from .interval import Interval

class TextGenerator(Base):
    def __init__(
        self, characters: str,
        textLength: int | Interval[int],
        allowRepetition: bool=True
    ):
        self.characters = characters
        self.textLength = textLength
        self.allowRepetition = allowRepetition
    
    def random(self) -> str:
        indices: list[int] = []
        characters = list(self.characters)
        textLength = self.textLength
        if type(textLength) is Interval:
            textLength = textLength.random()
        if self.allowRepetition:
            indices: list[int] = [random.randint(0, len(characters)-1) for i in range(textLength)]
        else:
            assert len(self.characters) >= textLength
            indices: list[int] = []
            availableIndices: list[int] = list(range(len(characters)))
            for i in range(textLength):
                j = random.randint(0, len(availableIndices)-1)
                indices.append(availableIndices[j])
                del availableIndices[j]
        return ''.join([characters[idx] for idx in indices])

    def debug():
        print(TextGenerator('abcdefghijklmnopqrstuvwxyz', 26, allowRepetition=False).random())

class TextSampler(Base):
    def __init__(self, textPopulation: list[str]):
        self.textPopulation = textPopulation
    
    def random(self) -> str:
        return random.sample(self.textPopulation, k=1)[0]

    def debug():
        sampler = TextSampler(StringSets.namae)
        for i in range(10):
            print(f"{sampler.random()=}")

ComposableText = str | TextGenerator | TextSampler

class TextComposer(BaseHandler[ComposableText]):
    def __init__(self, _objects: list[ComposableText]=None):
        super().__init__(_objects)
    
    def random(self) -> str:
        assert len(self) > 0
        result = ''
        for composableText in self:
            if type(composableText) is str:
                result += composableText
            else:
                result += composableText.random()
        return result

    def to_dict(self, **kwargs) -> dict:
        super().to_dict(compressed=False, **kwargs)

    def debug():
        numGen = TextGenerator('0123456789', textLength=Interval[int](1, 13))
        composer = TextComposer(['第', numGen, '号'])
        for i in range(10):
            print(composer.random())

class CharacterSets:
    alpha = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!#$%&()-=^~|@`[{;+:*]},<.>/?_"
    kana = "１２３４５６７８９０ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｈｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
    kanji = "亜哀挨愛曖悪握圧扱宛嵐安案暗以衣位囲医依委威為畏胃尉異移萎偉椅彙意違維慰遺緯域育一壱逸茨芋引印因咽姻員院淫陰飲隠韻右宇羽雨唄鬱畝浦運雲永泳英映栄営詠影鋭衛易疫益液駅悦越謁閲円延沿炎怨宴媛援園煙猿遠鉛塩演縁艶汚王凹央応往押旺欧殴桜翁奥横岡屋億憶臆虞乙俺卸音恩温穏下化火加可仮何花佳価果河苛科架夏家荷華菓貨渦過嫁暇禍靴寡歌箇稼課蚊牙瓦我画芽賀雅餓介回灰会快戒改怪拐悔海界皆械絵開階塊楷解潰壊懐諧貝外劾害崖涯街慨蓋該概骸垣柿各角拡革格核殻郭覚較隔閣確獲嚇穫学岳楽額顎掛潟括活喝渇割葛滑褐轄且株釜鎌刈干刊甘汗缶完肝官冠巻看陥乾勘患貫寒喚堪換敢棺款間閑勧寛幹感漢慣管関歓監緩憾還館環簡観韓艦鑑丸含岸岩玩眼頑顔願企伎危机気岐希忌汽奇祈季紀軌既記起飢鬼帰基寄規亀喜幾揮期棋貴棄毀旗器畿輝機騎技宜偽欺義疑儀戯擬犠議菊吉喫詰却客脚逆虐九久及弓丘旧休吸朽臼求究泣急級糾宮救球給嗅窮牛去巨居拒拠挙虚許距魚御漁凶共叫狂京享供協況峡挟狭恐恭胸脅強教郷境橋矯鏡競響驚仰暁業凝曲局極玉巾斤均近金菌勤琴筋僅禁緊錦謹襟吟銀区句苦駆具惧愚空偶遇隅串屈掘窟熊繰君訓勲薫軍郡群兄刑形系径茎係型契計恵啓掲渓経蛍敬景軽傾携継詣慶憬稽憩警鶏芸迎鯨隙劇撃激桁欠穴血決結傑潔月犬件見券肩建研県倹兼剣拳軒健険圏堅検嫌献絹遣権憲賢謙鍵繭顕験懸元幻玄言弦限原現舷減源厳己戸古呼固股虎孤弧故枯個庫湖雇誇鼓錮顧五互午呉後娯悟碁語誤護口工公勾孔功巧広甲交光向后好江考行坑孝抗攻更効幸拘肯侯厚恒洪皇紅荒郊香候校耕航貢降高康控梗黄喉慌港硬絞項溝鉱構綱酵稿興衡鋼講購乞号合拷剛傲豪克告谷刻国黒穀酷獄骨駒込頃今困昆恨根婚混痕紺魂墾懇左佐沙査砂唆差詐鎖座挫才再災妻采砕宰栽彩採済祭斎細菜最裁債催塞歳載際埼在材剤財罪崎作削昨柵索策酢搾錯咲冊札刷刹拶殺察撮擦雑皿三山参桟蚕惨産傘散算酸賛残斬暫士子支止氏仕史司四市矢旨死糸至伺志私使刺始姉枝祉肢姿思指施師恣紙脂視紫詞歯嗣試詩資飼誌雌摯賜諮示字寺次耳自似児事侍治持時滋慈辞磁餌璽鹿式識軸七𠮟失室疾執湿嫉漆質実芝写社車舎者射捨赦斜煮遮謝邪蛇尺借酌釈爵若弱寂手主守朱取狩首殊珠酒腫種趣寿受呪授需儒樹収囚州舟秀周宗拾秋臭修袖終羞習週就衆集愁酬醜蹴襲十汁充住柔重従渋銃獣縦叔祝宿淑粛縮塾熟出述術俊春瞬旬巡盾准殉純循順準潤遵処初所書庶暑署緒諸女如助序叙徐除小升少召匠床抄肖尚招承昇松沼昭宵将消症祥称笑唱商渉章紹訟勝掌晶焼焦硝粧詔証象傷奨照詳彰障憧衝賞償礁鐘上丈冗条状乗城浄剰常情場畳蒸縄壌嬢錠譲醸色拭食植殖飾触嘱織職辱尻心申伸臣芯身辛侵信津神唇娠振浸真針深紳進森診寝慎新審震薪親人刃仁尽迅甚陣尋腎須図水吹垂炊帥粋衰推酔遂睡穂16随髄枢崇数据杉裾寸瀬是井世正生成西声制姓征性青斉政星牲省凄逝清盛婿晴勢聖誠精製誓静請整醒税夕斥石赤昔析席脊隻惜戚責跡積績籍切折拙窃接設雪摂節説舌絶千川仙占先宣専泉浅洗染扇栓旋船戦煎羨腺詮践箋銭潜線遷選薦繊鮮全前善然禅漸膳繕狙阻祖租素措粗組疎訴塑遡礎双壮早争走奏相荘草送倉捜挿桑巣掃曹曽爽窓創喪痩葬装僧想層総遭槽踪操燥霜騒藻造像増憎蔵贈臓即束足促則息捉速側測俗族属賊続卒率存村孫尊損他多汰打妥唾堕惰駄太対体耐待怠胎退帯泰堆袋逮替貸隊滞態戴大代台第題滝宅択沢卓拓託濯諾濁但達脱奪棚誰丹旦担単炭胆探淡短嘆端綻誕鍛団男段断弾暖談壇地池知値恥致遅痴稚置緻竹畜逐蓄築秩窒茶着嫡中仲虫沖宙忠抽注昼柱衷酎鋳駐著貯丁弔庁兆町長挑帳張彫眺釣頂鳥朝12貼超腸跳徴嘲潮澄調聴懲直勅捗沈珍朕陳賃鎮追椎墜通痛塚漬坪爪鶴低呈廷弟定底抵邸亭貞帝訂庭逓停偵堤提程艇締諦泥的笛摘滴適敵溺迭哲鉄徹撤天典店点展添転塡田伝殿電斗吐妬徒途都渡塗賭土奴努度怒刀冬灯当投豆東到逃倒凍唐島桃討透党悼盗陶塔搭棟湯痘登答等筒統稲踏糖頭謄藤闘騰同洞胴動堂童道働銅導瞳峠匿特得督徳篤毒独読栃凸突届屯豚頓貪鈍曇丼那奈内梨謎鍋南軟難二尼弐匂肉虹日入乳尿任妊忍認寧熱年念捻粘燃悩納能脳農濃把波派破覇馬婆罵拝杯背肺俳配排敗廃輩売倍梅培陪媒買賠白伯拍泊迫剝舶博薄麦漠縛爆箱箸畑肌八鉢発髪伐抜罰閥反半氾犯帆汎伴判坂阪板版班畔般販斑飯搬煩頒範繁藩晩番蛮盤比皮妃否批彼披肥非卑飛疲秘被悲扉費碑罷避尾眉美備微鼻膝肘匹必泌筆姫百氷表俵票評漂標苗秒病描猫品浜貧賓頻敏瓶不夫父付布扶府怖阜附訃負赴浮婦符富普腐敷膚賦譜侮武部舞封風伏服副幅復福腹複覆払沸仏物粉紛雰噴墳憤奮分文聞丙平兵併並柄陛閉塀幣弊蔽餅米壁璧癖別蔑片辺返変偏遍編便勉歩保哺捕補舗母募墓慕暮簿方包芳邦奉宝抱放法泡胞俸倣峰砲崩訪報蜂豊飽褒縫亡乏忙坊妨忘防房肪某冒剖紡望傍帽棒貿貌暴膨謀頰北木朴牧睦僕墨撲没勃堀本奔翻凡盆麻摩磨魔毎妹枚昧埋幕膜枕又末抹万満慢漫未味魅岬密蜜脈妙民眠矛務無夢霧娘名命明迷冥盟銘鳴滅免面綿麺茂模毛妄盲耗猛網目黙門紋問4冶夜野弥厄役約訳薬躍闇由油喩愉諭輸癒唯友有勇幽悠郵湧猶裕遊雄誘憂融優与予余誉預幼用羊妖洋要容庸揚揺葉陽溶腰様瘍踊窯養擁謡曜抑沃浴欲翌翼拉裸羅来雷頼絡落酪辣乱卵覧濫藍欄吏利里理痢裏履璃離陸立律慄略柳流留竜粒隆硫侶旅虜慮了両良料涼猟陵量僚領寮療瞭糧力緑林厘倫輪隣臨瑠涙累塁類令礼冷励戻例鈴零霊隷齢麗暦歴列劣烈裂恋連廉練錬呂炉賂路露老労弄郎朗浪廊楼漏籠六録麓論和話賄脇惑枠湾腕"

class StringSets:
    namae = [
        "佐藤","鈴木","高橋","田中","伊藤","渡辺","山本","中村","小林","加藤","吉田","山田","佐々木",
        "山口","松本","井上","木村","林","斎藤","清水","山崎","森","池田","橋本","阿部","石川","山下",
        "中島","石井","小川","前田","岡田","長谷川","藤田","後藤","近藤","村上","遠藤","青木","坂本",
        "斉藤","福田","太田","西村","藤井","金子","岡本","藤原","中野","三浦","原田","中川","松田","竹内",
        "小野","田村","中山","和田","石田","森田","上田","原","内田","柴田","酒井","宮崎","横山","高木",
        "安藤","宮本","大野","小島","谷口","今井","工藤","高田","増田", "丸山","杉山","村田","大塚",
        "新井","小山","平野","藤本","河野","上野","野口","武田","松井","千葉","岩崎","菅原","木下","久保",
        "佐野","野村","松尾","市川","菊地","杉本","古川","大西","島田","水野","桜井","高野","渡部","吉川",
        "山内","西田","飯田","菊池","西川","小松","北村","安田","五十嵐","川口","平田","関","中田",
        "久保田","服部","東","岩田","土屋","川崎","福島","本田","辻","樋口","秋山","田口","永井","山中",
        "中西","吉村","川上","石原","大橋","松岡","馬場","浜田","森本","星野","矢野","浅野","大久保",
        "松下","吉岡","小池","野田","荒木","大谷","内藤","松浦","熊谷","黒田","尾崎","永田","川村","望月",
        "田辺","松村","荒井","堀","大島","平井","早川","菅野","栗原","西山","広瀬","横田","石橋","萩原",
        "岩本","片山","関口","宮田","大石","高山","本間","吉野","須藤","岡崎","小田","伊東","鎌田","上原",
        "篠原","小西","松原","福井","古賀","大森","小泉","成田","南","奥村","内山","沢田","川島","桑原",
        "三宅","片岡","富田","杉浦","岡","八木","奥田","小沢","松永","北川","関根","河合","平山","牧野",
        "白石","今村","寺田","青山","中尾","小倉","渋谷","上村","小野寺","大山","足立","岡村","坂口",
        "天野","多田","佐久間","根本","豊田","田島","飯塚","角田","村山","武藤","西","白井","竹田","宮下",
        "塚本","榎本","神谷","坂田","児玉","水谷","坂井","齋藤","小原","浅井","岡部","森下","小笠原",
        "神田","中井","植田","河村","宮川","稲垣","前川","大川","松崎","長田","若林","飯島","谷","大沢",
        "石塚","安部","堀内","及川","田代","中嶋","江口","山根","中谷","岸本","荒川","本多","西尾","森山",
        "岡野","金井","細川","今野","戸田","稲葉","安達","津田","森川","落合","土井","村松","星","町田",
        "三上","畠山","岩井","長尾","堤","中原","野崎","中沢","金田","米田","松山","杉田","堀田","西野",
        "三好","山岸","佐伯","黒川","西岡","泉","大竹","甲斐","笠原","大木","堀江","岸","徳永","川田",
        "須田","黒木","山川","古田","榊原","梅田","新田","三木","野中","大城","村井","奥山","金城","土田",
        "滝沢","大村","川端","井口","梶原","大場","宮城","長島","比嘉","吉原","宮内","金沢","安井","庄司",
        "大内","茂木","荻野","日高","松島","向井","下田","塚田","石黒","西本","奥野","竹中","広田","嶋田",
        "栗田","藤川","福本","北野","宇野","藤野","川原","谷川","丹羽","小谷","吉本","青柳","藤岡","竹本",
        "竹下","古谷","緒方","藤村","平川","亀井","高島","三輪","藤沢","篠崎","窪田","宮原","高井","根岸",
        "下村","高瀬","山村","川本","柳沢","横井","小澤","出口","吉沢","武井","小森","竹村","長野","宮沢",
        "志村","平松","臼井","福岡","黒沢","溝口","田原","稲田","浅田","筒井","柳田","奧村","大原",
        "永野","林田","冨田","大平","瀬戸","手塚","入江","篠田","福永","北原","富永","矢島","小出",
        "湯浅","鶴田","沼田","高松","長岡","堀口","岩瀬","山岡","大田","石崎","大槻","澤田","石山",
        "池上","堀川","二宮","相馬","園田","柏木","島崎","奧田","平岡","花田","杉原","加納","村瀬",
        "川野","片桐","内海","長沢","倉田","野沢","河原","福原","秋元","越智","西原","松野","笠井",
        "小坂","田畑","日野","北島","渡邊","谷本","千田","吉井","深沢","西沢","相沢","徳田","原口",
        "小柳","米山","新谷","細谷","田上","今泉","菅","浜野","森岡"
    ]