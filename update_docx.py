import docx

def update_docx(file_path):
    doc = docx.Document(file_path)
    
    for p in doc.paragraphs:
        # PENTING: Perbaiki aturan preprocessing yang keliru
        if 'resize (shortest side = 256, jaga rasio)' in p.text:
            p.text = p.text.replace('resize (shortest side = 256, jaga rasio)', 'resize langsung ke ukuran 224x224 (simple resize, tidak perlu preserve aspect ratio)')
        if 'center crop ke' in p.text:
            p.text = p.text.replace('center crop ke 224x224', '(PENTING: Jangan gunakan center crop! Ini akan merusak orientasi retakan dan membuat model salah menebak kelas Vertikal)')
        
        # Opsi lain (mungkin ada simbol yang beda atau encoding)
        if 'resize (shortest' in p.text:
            p.text = p.text.replace(p.text, p.text.replace('resize (shortest', 'resize langsung (simple resize) ke 224x224'))
            
        if 'center crop' in p.text.lower():
            p.text = 'PENTING: Jangan gunakan center crop! Cukup gunakan simple resize ke 224x224.'
            
    doc.save('how_to_use_mobile_updated.docx')
    print('Docx successfully updated and saved as how_to_use_mobile_updated.docx')

update_docx('how_to_use_mobile.docx')
