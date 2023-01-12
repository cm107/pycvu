from pycvu.util.pdf import PDF

path = "/home/clayton/Desktop/35206365.pdf"
doc = PDF(path)

page = doc[0]
page.dpi = (256, 256)
print(f"{page.get_text()=}")
img = page.get_image()
img.save('/home/clayton/Desktop/35206365.png')
img = page.draw_rect(
    img=img,
    rects=page.search_for('井上', hit_max=15)
)
img.show()
