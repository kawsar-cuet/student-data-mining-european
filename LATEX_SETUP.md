# LaTeX Setup Guide for VS Code

## ✅ Step 1: LaTeX Workshop Extension (DONE)
The LaTeX Workshop extension has been installed in VS Code.

## 📦 Step 2: Install MiKTeX (LaTeX Distribution)

You need to install MiKTeX to actually compile LaTeX files. Choose ONE method:

### Option A: Using Windows Package Manager (Recommended)
1. **Open PowerShell as Administrator** (Right-click Start → Windows PowerShell (Admin))
2. Run this command:
   ```powershell
   winget install MiKTeX.MiKTeX -e --accept-package-agreements --accept-source-agreements
   ```
3. After installation completes, **close and reopen VS Code**

### Option B: Manual Download
1. Download MiKTeX from: https://miktex.org/download
2. Run the installer (choose "Install MiKTeX for all users" if you have admin rights)
3. During installation, set "Install missing packages" to **"Yes"** or **"Ask me first"**
4. After installation, **close and reopen VS Code**

## 🔧 Step 3: Verify Installation

After installing MiKTeX and reopening VS Code:

1. Open the terminal in VS Code (Ctrl + `)
2. Run: `pdflatex --version`
3. You should see MiKTeX version information

## 📝 Step 4: Compile Your LaTeX Document

1. Open `docs/JOURNAL_METHODOLOGY.tex` in VS Code
2. Press **Ctrl + Alt + B** (or click the green play button in the top right)
3. The PDF will be generated automatically!

### Alternative: Use the LaTeX Workshop sidebar
- Click the "TeX" icon in the left sidebar
- Under "Build LaTeX project", click "Recipe: pdflatex ➞ bibtex ➞ pdflatex × 2"

## 🎯 Configuration (DONE)

LaTeX Workshop has been configured in `.vscode/settings.json` with:
- ✅ Auto-build on save
- ✅ Auto-clean auxiliary files after build
- ✅ PDF viewer in VS Code tab
- ✅ Standard compilation recipes (pdflatex + bibtex)

## 📚 Useful Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + Alt + B` | Build LaTeX project |
| `Ctrl + Alt + V` | View PDF |
| `Ctrl + Alt + J` | Jump to location in PDF |
| `Ctrl + Click` | Navigate in PDF (SyncTeX) |

## ⚠️ Common Issues

### "pdflatex not found"
- Make sure MiKTeX is installed
- Restart VS Code after installation
- Check if `C:\Program Files\MiKTeX\miktex\bin\x64\` is in your PATH

### Missing packages
- First compilation may be slow as MiKTeX downloads required packages
- If prompted, allow MiKTeX to install missing packages automatically

### Compilation errors
- Check the "Problems" panel (Ctrl + Shift + M)
- View full log in "Output" panel → "LaTeX Workshop" channel

## 🚀 Next Steps

Once MiKTeX is installed:
1. Open `JOURNAL_METHODOLOGY.tex`
2. Press `Ctrl + Alt + B`
3. Your PDF will appear in `docs/JOURNAL_METHODOLOGY.pdf`

All tables and figures will be properly rendered in the PDF!
