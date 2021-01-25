from IPython.display import HTML 
from base64 import b64encode

def render_mp4(filepath, size=(300,300)):
    with open(filepath,'rb') as f:
        mp4 = f.read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

    html = HTML(f"""
                <video width="{size[0]}" height="{size[1]}" controls>
                <source src="{data_url}" type="video/mp4">
                </video>
                """)
    return html