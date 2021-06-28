SAVE_DIRECTORY = "/media/ssd/christen-rnd/Experiments/Lip2Speech/Datasets/DL"

VIDEO_IDS = """
4gm4sXPndUY
DirgOcmERwA
SSpSk1Io52w
_z27qjHjqiM
wPz3MPl5jvY
TtU9GhLeOk4
05L1tSvzWHs
VRcixOuG-TU
XKnBn1XZpQI
sbvv-uQmwVY
CJr4Dst0uZE
74clhgHhR2M
WpR8eOLUo9Q
DaLZRH6D7wo
a4k8YorzUdI
mdZVysybPrk
_GRvjpJVr5A
KHMHOieuJuw
XE8O5p5mceA
vAOI9kTDVoo
t5Q2z-MM1X4
BW7P1fvnAWk
FsWX2c7Q6Qg
mHhhXlEB32c
-7l-Pl_pwu4
AjUgxQNZhck
SPJnlFtw6h0
1pgahpCZeF0
0DKOUFrP7xI
lg4OLAjxRcQ
6USgwLa-7ks
mNUAt7umSiw
HxKaLyyGq50
gyGMaNUaeic
ewN0vFYFJ7A
zgfcRJYA_kk
DpK_i6iA0i0
gZ65vyTNipM
Xeb6OjnVn8g
zA0U8C_q04M
aPfkYu_qiF4
JSfRBOWp3Es
EZ4X5rFzLFE
YHAIrwRVvWg
CxgYyBut6i0
aVAs2YwiDoA
HnaBcIL2shc
pI-3TreYU18
yInilk6x-OY
mvLBBG7Kvn8
qfsacbIe9AI
PmZp5VtMwLE
b0M0AzjpPqc
joKs2EJ9Z8w
qlNDzNjR3eA
4TC5s_xNKSs
BTzz3BuYuM8
dC_d3iVw6sc
5eEPs8XS5AQ
Y0m136XU65o
EB1SoyivHFU
Vc-jG_LdOLw
1XMjfhEFbFA
yBmtXtVya9A
yw8xwS15Pf4
giZD8yzXEZ4
agGUR06jM_g
UukbB6q6CTo
53ysOFHCaqI
_EwqeZcZ9KY
MBcnPueElDU
zm5cqvfKO-o
AASR9rOzhhA
3F6KInYCJxI
_BBTTS9gx7Q
rcXOEzAuvu4
PMv8C-Ws1b8
Qdh4NEkQRuo
FKCV76N9Ys0
H_ptjlq8pH0
rSY5YLLcPs8
uDcU3ZzH7hs
dKKPtG-qeaI
f8tod7wt2Vw
9eZN8U-68eI
X4RmokyD3U8
hvhqHhrP_AU
mvljKZtGTj4
gctnJbUaW08
XO2GHDm-ZoI
qQmMp8fL82w
M2T36PlRv1U
0n2x_D-ZmmU
pHarJN7Wo0M
hMkRDNzH9vs
orTNf_R5odY
XZUGn36eAgA
nBxy_vzmO_M
tcR5I2pPk3o
0OHLbVEOq04
6XhSJbfT1pk
ioe1eeEWU0I
0ZQxPIwuA4o
7Np_5Q-P8eo
y_tQBokE94s
zqbgsiKBSRI
wHHxkWcqokY
-0ZMU-gnm2g
xV1Q3go0S1Y
REzrCEDMQws
ym5qG-3kJ10
7eiYBFladus
sV9aiEsXanE
""".split('\n')[1:-1]


import youtube_dl


video_links = list()
for yt_id in VIDEO_IDS:
    link = f'https://www.youtube.com/watch?v={yt_id}'
    video_links.append(link)

opts = {
    'outtmpl': f'{SAVE_DIRECTORY}/%(title)s-%(id)s.%(ext)s',
    'format': '22/18'
}
with youtube_dl.YoutubeDL(opts) as ydl:
    ydl.download(video_links)
