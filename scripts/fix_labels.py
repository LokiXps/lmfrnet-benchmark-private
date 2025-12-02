import json

# Current LABELS from index.html (with some known fixes needed)
current_labels = [
    { "en": "Faces", "ja": "顔" }, { "en": "Faces_easy", "ja": "顔(Easy)" },
    { "en": "Leopards", "ja": "ヒョウ" }, { "en": "Motorbikes", "ja": "バイク" }, { "en": "accordion", "ja": "アコーディオン" },
    { "en": "airplane", "ja": "飛行機" }, { "en": "anchor", "ja": "錨" }, { "en": "ant", "ja": "アリ" },
    { "en": "barrel", "ja": "樽" }, { "en": "bass", "ja": "バス(魚)" }, { "en": "beaver", "ja": "ビーバー" },
    { "en": "binocular", "ja": "双眼鏡" }, { "en": "bonsai", "ja": "盆栽" }, { "en": "brain", "ja": "脳" },
    { "en": "brontosaurus", "ja": "ブロントサウルス" }, { "en": "buddha", "ja": "仏像" }, { "en": "butterfly", "ja": "蝶" },
    { "en": "camera", "ja": "カメラ" }, { "en": "cannon", "ja": "大砲" }, { "en": "car_side", "ja": "車(横)" },
    { "en": "ceiling_fan", "ja": "シーリングファン" }, { "en": "cellphone", "ja": "携帯電話" }, { "en": "chair", "ja": "椅子" },
    { "en": "chandelier", "ja": "シャンデリア" }, { "en": "cougar_body", "ja": "クーガー(体)" }, { "en": "cougar_face", "ja": "クーガー(顔)" },
    { "en": "crab", "ja": "カニ" }, { "en": "crayfish", "ja": "ザリガニ" }, { "en": "crocodile", "ja": "ワニ" },
    { "en": "cup", "ja": "カップ" }, { "en": "dalmatian", "ja": "ダルメシアン" }, { "en": "dollar_bill", "ja": "ドル札" },
    { "en": "dolphin", "ja": "イルカ" }, { "en": "dragonfly", "ja": "トンボ" }, { "en": "electric_guitar", "ja": "エレキギター" },
    { "en": "elephant", "ja": "象" }, { "en": "emu", "ja": "エミュー" }, { "en": "euphonium", "ja": "ユーフォニアム" },
    { "en": "ewer", "ja": "水差し" }, { "en": "ferry", "ja": "フェリー" }, { "en": "flamingo", "ja": "フラミンゴ" },
    { "en": "flamingo_head", "ja": "フラミンゴ(頭)" }, { "en": "garfield", "ja": "ガーフィールド" }, { "en": "gerenuk", "ja": "ゲレヌク" },
    { "en": "gramophone", "ja": "蓄音機" }, { "en": "grand_piano", "ja": "グランドピアノ" }, { "en": "hawksbill", "ja": "タイマイ(亀)" },
    { "en": "headphone", "ja": "ヘッドフォン" }, { "en": "hedgehog", "ja": "ハリネズミ" }, { "en": "helicopter", "ja": "ヘリコプター" },
    { "en": "ibis", "ja": "トキ" }, { "en": "inline_skate", "ja": "インラインスケート" }, { "en": "joshua_tree", "ja": "ジョシュアツリー" },
    { "en": "kangaroo", "ja": "カンガルー" }, { "en": "ketch", "ja": "ケッチ(帆船)" }, { "en": "lamp", "ja": "ランプ" },
    { "en": "laptop", "ja": "ノートPC" }, { "en": "llama", "ja": "ラマ" }, { "en": "lobster", "ja": "ロブスター" },
    { "en": "lotus", "ja": "蓮" }, { "en": "mandolin", "ja": "マンドリン" }, { "en": "mayfly", "ja": "カゲロウ" },
    { "en": "menorah", "ja": "メノラー" }, { "en": "metronome", "ja": "メトロノーム" }, { "en": "minaret", "ja": "ミナレット" },
    { "en": "nautilus", "ja": "オウムガイ" }, { "en": "octopus", "ja": "タコ" }, { "en": "okapi", "ja": "オカピ" },
    { "en": "pagoda", "ja": "塔" }, { "en": "panda", "ja": "パンダ" }, { "en": "pigeon", "ja": "鳩" },
    { "en": "pizza", "ja": "ピザ" }, { "en": "platypus", "ja": "カモノハシ" }, { "en": "pyramid", "ja": "ピラミッド" },
    { "en": "revolver", "ja": "リボルバー" }, { "en": "rhino", "ja": "サイ" }, { "en": "rooster", "ja": "雄鶏" },
    { "en": "saxophone", "ja": "サックス" }, { "en": "schooner", "ja": "スクーナー(船)" }, { "en": "scissors", "ja": "ハサミ" },
    { "en": "scorpion", "ja": "サソリ" }, { "en": "sea_horse", "ja": "タツノオトシゴ" }, { "en": "snoopy", "ja": "スヌーピー" },
    { "en": "soccer_ball", "ja": "サッカーボール" }, { "en": "stapler", "ja": "ホッチキス" }, { "en": "starfish", "ja": "ヒトデ" },
    { "en": "stegosaurus", "ja": "ステゴサウルス" }, { "en": "stop_sign", "ja": "一時停止標識" }, { "en": "strawberry", "ja": "イチゴ" },
    { "en": "sunflower", "ja": "ヒマワリ" }, { "en": "tick", "ja": "ダニ" }, { "en": "trilobite", "ja": "三葉虫" },
    { "en": "umbrella", "ja": "傘" }, { "en": "watch", "ja": "腕時計" }, { "en": "water_lilly", "ja": "スイレン" },
    { "en": "wheelchair", "ja": "車椅子" }, { "en": "wild_cat", "ja": "ヤマネコ" }, { "en": "windsor_chair", "ja": "ウィンザーチェア" },
    { "en": "wrench", "ja": "レンチ" }, { "en": "yin_yang", "ja": "陰陽" }
]

# Create mapping dictionary
trans_map = { item["en"]: item["ja"] for item in current_labels }
# Fix known discrepancies
trans_map["airplanes"] = "飛行機" # index.html has "airplane"
trans_map["crocodile_head"] = "ワニ(頭)" # missing in index.html

# Load true labels
with open('/home/loki/public/labels.json', 'r') as f:
    true_labels = json.load(f)

# Generate new LABELS list
new_labels = []
for label in true_labels:
    ja = trans_map.get(label, label) # Use English if no translation found
    new_labels.append({ "en": label, "ja": ja })

# Output as JS code
print("const LABELS = [")
for i, item in enumerate(new_labels):
    comma = "," if i < len(new_labels) - 1 else ""
    print(f'    {{ en: "{item["en"]}", ja: "{item["ja"]}" }}{comma}')
print("];")
