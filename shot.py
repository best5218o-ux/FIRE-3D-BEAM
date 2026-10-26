import sys, asyncio
from playwright.async_api import async_playwright
async def main(html, out, actions):
    async with async_playwright() as p:
        b = await p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
        pg = await b.new_page(viewport={'width': 1600, 'height': 900})
        errs = []
        pg.on('console', lambda m: errs.append(m.text) if m.type in ('error', 'warning') else None)
        pg.on('pageerror', lambda e: errs.append('PAGEERR ' + str(e)))
        await pg.goto('file://' + html); await pg.wait_for_timeout(2500); print('LOAD', errs[:5])
        for i, a in enumerate(actions):
            if a: await pg.evaluate(a); await pg.wait_for_timeout(1800)
            await pg.screenshot(path=f'{out}_{i}.png')
        print('\n'.join(errs[:10]))
        await b.close()
acts = sys.argv[3:] or ['']
asyncio.run(main(sys.argv[1], sys.argv[2], acts))
