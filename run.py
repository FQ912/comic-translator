import ssl
import os
import sys
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Отключаем SSL для скачивания моделей
ssl._create_default_https_context = ssl._create_unverified_context

# Импорты внешних библиотек
import easyocr
import cv2
from PIL import Image, ImageDraw, ImageFont
from googletrans import Translator


# ============================================================
# 1. Модели данных (Data Classes)
# ============================================================

@dataclass
class TextRegion:
    """Область текста на изображении"""
    text: str
    x: int
    y: int
    width: int
    height: int


@dataclass
class TranslatedRegion(TextRegion):
    """Область с переведённым текстом"""
    translated_text: str


# ============================================================
# 2. Интерфейсы (ISP - Interface Segregation Principle)
# ============================================================

class OCRProvider(ABC):
    """Абстракция для OCR движка (DIP)"""
    @abstractmethod
    def recognize(self, image_path: str) -> List[TextRegion]:
        pass


class TranslatorProvider(ABC):
    """Абстракция для переводчика (DIP)"""
    @abstractmethod
    def translate(self, text: str, src: str, dest: str) -> str:
        pass


class ImageProcessor(ABC):
    """Абстракция для обработки изображений (ISP)"""
    @abstractmethod
    def draw_text(self, image, regions: List[TranslatedRegion]) -> Image.Image:
        pass


# ============================================================
# 3. Реализации (OCP - Open/Closed Principle)
# ============================================================

class EasyOCRProvider(OCRProvider):
    """Реализация OCR через EasyOCR"""
    
    def __init__(self):
        self._reader = None
    
    def _get_reader(self):
        if self._reader is None:
            print("📦 Загрузка модели OCR...")
            self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._reader
    
    def recognize(self, image_path: str) -> List[TextRegion]:
        reader = self._get_reader()
        img = cv2.imread(image_path)
        results = reader.readtext(img, detail=1)
        
        regions = []
        for bbox, text, confidence in results:
            if confidence > 0.3 and len(text) > 1:
                x_coords = [p[0] for p in bbox]
                y_coords = [p[1] for p in bbox]
                
                regions.append(TextRegion(
                    text=text,
                    x=int(min(x_coords)),
                    y=int(min(y_coords)),
                    width=int(max(x_coords) - min(x_coords)),
                    height=int(max(y_coords) - min(y_coords))
                ))
        
        return self._group_regions(regions)
    
    def _group_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Группирует близкие регионы"""
        if not regions:
            return []
        
        regions.sort(key=lambda r: (r.y, r.x))
        grouped = []
        current = [regions[0]]
        
        for region in regions[1:]:
            if abs(region.y - current[-1].y) < 40:
                current.append(region)
            else:
                grouped.append(self._merge_regions(current))
                current = [region]
        
        if current:
            grouped.append(self._merge_regions(current))
        
        return grouped
    
    def _merge_regions(self, regions: List[TextRegion]) -> TextRegion:
        """Объединяет несколько регионов в один"""
        min_x = min(r.x for r in regions)
        min_y = min(r.y for r in regions)
        max_x = max(r.x + r.width for r in regions)
        max_y = max(r.y + r.height for r in regions)
        full_text = ' '.join(r.text for r in regions)
        
        return TextRegion(
            text=full_text,
            x=min_x - 10,
            y=min_y - 10,
            width=max_x - min_x + 20,
            height=max_y - min_y + 20
        )


class GoogleTranslateProvider(TranslatorProvider):
    """Реализация перевода через Google Translate"""
    
    def __init__(self):
        self._translator = Translator()
    
    def translate(self, text: str, src: str = 'en', dest: str = 'ru') -> str:
        try:
            result = self._translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception:
            return text


class PILImageProcessor(ImageProcessor):
    """Реализация обработки изображений через PIL"""
    
    def __init__(self, font_size: int = 16):
        self.font_size = font_size
        self._font = None
    
    def _get_font(self):
        if self._font is None:
            try:
                self._font = ImageFont.truetype("arial.ttf", self.font_size)
            except:
                self._font = ImageFont.load_default()
        return self._font
    
    def draw_text(self, image, regions: List[TranslatedRegion]) -> Image.Image:
        result = image.copy()
        draw = ImageDraw.Draw(result)
        font = self._get_font()
        
        for region in regions:
            # Очищаем область
            draw.rectangle([
                region.x, region.y,
                region.x + region.width, region.y + region.height
            ], fill='white')
            
            # Разбиваем текст на строки
            lines = self._wrap_text(region.translated_text, region.width)
            
            # Рисуем по центру
            y_offset = region.y + (region.height - len(lines) * 22) // 2
            for line in lines:
                line_width = len(line) * 10
                x_offset = region.x + (region.width - line_width) // 2
                draw.text((x_offset, y_offset), line, fill='black', font=font)
                y_offset += 22
        
        return result
    
    def _wrap_text(self, text: str, max_width: int) -> List[str]:
        """Переносит текст на новые строки"""
        words = text.split()
        lines = []
        current = []
        
        for word in words:
            current.append(word)
            if len(' '.join(current)) * 10 > max_width - 20:
                current.pop()
                if current:
                    lines.append(' '.join(current))
                current = [word]
        
        if current:
            lines.append(' '.join(current))
        
        return lines


# ============================================================
# 4. Оркестратор (SRP - Single Responsibility)
# ============================================================

class ComicTranslator:
    """Главный класс, координирующий работу (Facade pattern)"""
    
    def __init__(self, ocr: OCRProvider, translator: TranslatorProvider, image_processor: ImageProcessor):
        self.ocr = ocr
        self.translator = translator
        self.image_processor = image_processor
    
    def translate(self, input_path: str, output_path: str) -> int:
        """Переводит комикс и сохраняет результат"""
        print(f"\n📖 Обработка: {os.path.basename(input_path)}")
        
        # 1. Распознаём текст
        regions = self.ocr.recognize(input_path)
        
        if not regions:
            print("   ⚠️ Текст не найден")
            return 0
        
        # 2. Переводим
        translated_regions = []
        for region in regions:
            translated_text = self.translator.translate(region.text)
            translated_regions.append(TranslatedRegion(
                text=region.text,
                translated_text=translated_text,
                x=region.x, y=region.y,
                width=region.width, height=region.height
            ))
            print(f"   {region.text[:30]} → {translated_text[:30]}")
        
        # 3. Отрисовываем результат
        original_image = Image.open(input_path).convert('RGB')
        result_image = self.image_processor.draw_text(original_image, translated_regions)
        result_image.save(output_path)
        
        print(f"   ✅ Сохранено: {output_path}")
        return len(translated_regions)


# ============================================================
# 5. Файловые операции и меню (KISS - Keep It Simple)
# ============================================================

class FileManager:
    """Управление файлами"""
    
    @staticmethod
    def ensure_directories():
        """Создаёт необходимые папки"""
        os.makedirs('to_translate', exist_ok=True)
        os.makedirs('translated', exist_ok=True)
    
    @staticmethod
    def scan_images() -> List[str]:
        """Сканирует папку to_translate на наличие изображений"""
        extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
        files = []
        for f in os.listdir('to_translate'):
            if f.lower().endswith(extensions):
                files.append(f)
        return files


class ConsoleMenu:
    """Интерактивное меню"""
    
    def __init__(self, translator: ComicTranslator):
        self.translator = translator
    
    def run(self):
        """Запускает меню"""
        FileManager.ensure_directories()
        
        while True:
            files = FileManager.scan_images()
            
            if not files:
                print("\n⚠️ В папке 'to_translate' нет файлов!")
                break
            
            self._show_menu()
            choice = input("Выберите действие: ").strip()
            
            if choice == '0':
                print("До свидания!")
                break
            elif choice == '1':
                self._translate_all(files)
            elif choice == '2':
                self._translate_selected(files)
            elif choice == '3':
                self._show_file_list(files)
            else:
                print("❌ Неверный выбор!")
            
            input("\nНажмите Enter для продолжения...")
    
    def _show_menu(self):
        print("\n" + "="*60)
        print("ГЛАВНОЕ МЕНЮ")
        print("="*60)
        print("1. Перевести ВСЕ картинки")
        print("2. Выбрать картинки (по номерам через пробел)")
        print("3. Показать список файлов")
        print("0. Выход")
        print("="*60)
    
    def _translate_all(self, files: List[str]):
        """Переводит все файлы"""
        print(f"\n🔄 Переводим {len(files)} файлов...")
        for i, file in enumerate(files, 1):
            input_path = os.path.join('to_translate', file)
            output_path = os.path.join('translated', f"translated_{file}")
            print(f"\n[{i}/{len(files)}]", end="")
            self.translator.translate(input_path, output_path)
        print(f"\n✅ Готово!")
    
    def _translate_selected(self, files: List[str]):
        """Переводит выбранные файлы"""
        print("\nВыберите картинки (введите номера через пробел):")
        print("   Например: 1 3 5")
        print()
        for i, f in enumerate(files, 1):
            print(f"   {i}. {f}")
        
        nums = input("\nНомера: ").strip()
        if not nums:
            return
        
        selected = []
        for num in nums.split():
            try:
                idx = int(num) - 1
                if 0 <= idx < len(files):
                    selected.append(files[idx])
            except:
                pass
        
        if selected:
            print(f"\n🔄 Переводим {len(selected)} файлов...")
            for i, file in enumerate(selected, 1):
                input_path = os.path.join('to_translate', file)
                output_path = os.path.join('translated', f"translated_{file}")
                print(f"\n[{i}/{len(selected)}]", end="")
                self.translator.translate(input_path, output_path)
            print(f"\n✅ Готово!")
    
    def _show_file_list(self, files: List[str]):
        """Показывает список файлов"""
        print(f"\n📁 Файлы в папке 'to_translate' ({len(files)} шт.):")
        for i, f in enumerate(files, 1):
            size = os.path.getsize(os.path.join('to_translate', f)) / 1024
            print(f"   {i}. {f} ({size:.1f} KB)")


# ============================================================
# 6. Точка входа
# ============================================================

def main():
    """Главная функция"""
    print("="*60)
    print("ПЕРЕВОДЧИК КОМИКСОВ v5.0 (SOLID)")
    print("="*60)
    
    # Создаём зависимости (Dependency Injection)
    ocr = EasyOCRProvider()
    translator = GoogleTranslateProvider()
    image_processor = PILImageProcessor()
    
    # Создаём оркестратор
    comic_translator = ComicTranslator(ocr, translator, image_processor)
    
    # Запускаем меню или обрабатываем аргументы
    if len(sys.argv) > 1:
        # Режим командной строки
        FileManager.ensure_directories()
        cmd = sys.argv[1]
        
        if cmd == 'all':
            files = FileManager.scan_images()
            for i, file in enumerate(files, 1):
                input_path = os.path.join('to_translate', file)
                output_path = os.path.join('translated', f"translated_{file}")
                comic_translator.translate(input_path, output_path)
        elif cmd == 'list':
            files = FileManager.scan_images()
            for i, f in enumerate(files, 1):
                print(f"{i}. {f}")
        elif cmd.isdigit():
            files = FileManager.scan_images()
            idx = int(cmd) - 1
            if 0 <= idx < len(files):
                file = files[idx]
                input_path = os.path.join('to_translate', file)
                output_path = os.path.join('translated', f"translated_{file}")
                comic_translator.translate(input_path, output_path)
    else:
        # Интерактивный режим
        menu = ConsoleMenu(comic_translator)
        menu.run()


if __name__ == "__main__":
    main()