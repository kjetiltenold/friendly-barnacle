import base64
import unittest
from unittest.mock import patch

from app.attachments.parser import _process_pdf, process_attachments
from app.models import FileAttachment


class _FakePixmap:
    def tobytes(self, fmt: str) -> bytes:
        if fmt != "png":
            raise AssertionError(f"Unexpected pixmap format {fmt}")
        return b"png-bytes"


class _FakePage:
    def __init__(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text

    def get_pixmap(self, dpi: int):
        if dpi != 200:
            raise AssertionError(f"Unexpected dpi {dpi}")
        return _FakePixmap()


class _FakeDoc(list):
    def close(self) -> None:
        return None


class AttachmentParserTests(unittest.TestCase):
    def test_process_attachments_parses_csv_into_text_block(self):
        csv_bytes = "Dato;Belop;Tekst\n2026-02-15;24000;Nordlicht GmbH\n2026-02-15;-14500;Arctic Supplies\n".encode("utf-8")
        attachment = FileAttachment(
            filename="bank.csv",
            mime_type="text/csv",
            content_base64=base64.b64encode(csv_bytes).decode("ascii"),
        )

        blocks = process_attachments([attachment])

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertIn("[Content from bank.csv]", blocks[0]["text"])
        self.assertIn("Parsed CSV with 3 row(s).", blocks[0]["text"])
        self.assertIn("Dato\tBelop\tTekst", blocks[0]["text"])
        self.assertIn("2026-02-15\t24000\tNordlicht GmbH", blocks[0]["text"])

    def test_process_pdf_adds_image_for_short_text_page(self):
        doc = _FakeDoc([_FakePage("NSB\nTogbillett\n13.04.2026\n141,00\n")])

        with patch("app.attachments.parser.fitz.open", return_value=doc):
            blocks = _process_pdf(b"%PDF", "receipt.pdf")

        self.assertEqual(blocks[0]["type"], "image")
        self.assertEqual(blocks[1]["type"], "text")
        self.assertIn("OCR text may miss separators", blocks[1]["text"])

    def test_process_pdf_adds_image_for_single_page_contract(self):
        long_text = "\n".join(
            [
                "Arbeidskontrakt mellom selskapet og arbeidstakeren med utfyllende beskrivelser av rolle, ansvar, arbeidssted og praktiske forhold for ansettelsen.",
                "Stillingen gjelder fast ansettelse i utviklingsavdelingen, og kontrakten beskriver forventninger til leveranser, samarbeid, ferie, pensjon, forsikring og oppsigelsestid i detalj.",
                "Lonn, stillingsprosent, arbeidssted, rapporteringslinjer og standard arbeidstid er spesifisert nedenfor sammen med kontaktpersoner, interne rutiner og administrative opplysninger.",
                "Kontrakten starter 2026-07-25 og har vanlige vilkar, med flere avsnitt om taushetsplikt, bruk av utstyr, reiser, hjemmekontor og andre relevante forhold i arbeidsforholdet.",
                "Dette er nok tekst til at modellen kan lese dokumentet uten et ekstra bilde, fordi siden inneholder lange, tydelige tekstlinjer og ikke ligner en kort kvittering eller en svak OCR-side.",
            ]
        )
        doc = _FakeDoc([_FakePage(long_text)])

        with patch("app.attachments.parser.fitz.open", return_value=doc):
            blocks = _process_pdf(b"%PDF", "contract.pdf")

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["type"], "text")
        self.assertEqual(blocks[1]["type"], "image")

    def test_process_pdf_skips_images_for_multi_page_long_text_document(self):
        long_text = "\n".join(
            [
                "Arbeidskontrakt mellom selskapet og arbeidstakeren med utfyllende beskrivelser av rolle, ansvar, arbeidssted og praktiske forhold for ansettelsen.",
                "Stillingen gjelder fast ansettelse i utviklingsavdelingen, og kontrakten beskriver forventninger til leveranser, samarbeid, ferie, pensjon, forsikring og oppsigelsestid i detalj.",
                "Lonn, stillingsprosent, arbeidssted, rapporteringslinjer og standard arbeidstid er spesifisert nedenfor sammen med kontaktpersoner, interne rutiner og administrative opplysninger.",
                "Kontrakten starter 2026-07-25 og har vanlige vilkar, med flere avsnitt om taushetsplikt, bruk av utstyr, reiser, hjemmekontor og andre relevante forhold i arbeidsforholdet.",
                "Dette er nok tekst til at modellen kan lese dokumentet uten et ekstra bilde, fordi siden inneholder lange, tydelige tekstlinjer og ikke ligner en kort kvittering eller en svak OCR-side.",
            ]
        )
        doc = _FakeDoc([_FakePage(long_text), _FakePage(long_text)])

        with patch("app.attachments.parser.fitz.open", return_value=doc):
            blocks = _process_pdf(b"%PDF", "contract.pdf")

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["type"], "text")


if __name__ == "__main__":
    unittest.main()
