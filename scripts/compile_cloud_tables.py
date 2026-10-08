import pathlib,sys,json,subprocess
root=pathlib.Path(sys.argv[1]).resolve();out=root/'docs/cloud-reproduction/latex-proof';out.mkdir(exist_ok=True)
manifest=json.loads((root/'docs/cloud-reproduction/publication-manifest.json').read_text());files=[root/x['path'] for x in manifest if x['path'].endswith('.tex')]
lines=[r'\documentclass{article}',r'\usepackage[paperwidth=18in,paperheight=24in,margin=1in]{geometry}',r'\usepackage{booktabs,graphicx}',r'\begin{document}']
for p in files:lines += [r'\section*{'+p.stem.replace('_',r'\_')+'}',r'\resizebox{\linewidth}{!}{\input{'+str(p.relative_to(root))+'}}',r'\clearpage']
lines += [r'\end{document}'];(out/'cloud-publication-proof.tex').write_text('\n'.join(lines)+'\n')
r=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),str(out/'cloud-publication-proof.tex')],cwd=root,capture_output=True,text=True);(out/'compiler-stdout.txt').write_text(r.stdout+r.stderr);assert r.returncode==0,'Inspect compiler-stdout.txt'
(out/'compile.json').write_text(json.dumps(dict(passed=True,engine=subprocess.check_output(['pdflatex','--version'],text=True).splitlines()[0],tables=[str(p.relative_to(root)) for p in files],pdf='docs/cloud-reproduction/latex-proof/cloud-publication-proof.pdf'),indent=2)+'\n')
(out/'cloud-publication-proof.aux').unlink(missing_ok=True);print('Compiled',len(files),'current tables')
