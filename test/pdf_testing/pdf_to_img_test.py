from pycvu.util.pdf import pdf_to_png

pdf_to_png(
    path='/home/clayton/workspace/prj/data/MediaTech/20230106/DTFS_AIOCRテストデータ',
    output_dir='dump',
    pool=16, save_cpus=2,
    background_color='white'
)