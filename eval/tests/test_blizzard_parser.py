"""Table-driven tests for the Blizzard filename parser — every year's pattern."""

import pytest
from srm_eval.data.blizzard import parse_filepath

# Years 2008-2023: (year, filepath, expected_subtask, expected_system)
PARSE_CASES = [
    # 2008: lang_corpus encoding, no track column.
    (2008, "Blizzard_2008/A_submission_directory_english_arctic_2008_news_news_2008_0002.wav", "english_arctic", "A"),
    (2008, "Blizzard_2008/B_submission_directory_english_arctic_2008_news_news_2008_0002.wav", "english_arctic", "B"),
    (2008, "Blizzard_2008/C_submission_directory_english_arctic_2008_news_news_2008_0002.wav", "english_arctic", "C"),
    (2008, "Blizzard_2008/N_submission_directory_mandarin_arctic_news_news_2008_0088.wav", "mandarin_arctic", "N"),
    (2008, "Blizzard_2008/R_submission_directory_english_rjs_2008_news_news_2008_0002.wav", "english_rjs", "R"),

    # 2009: lang_task encoding.
    (2009, "Blizzard_2009/A_submission_directory_english_EH1_2009_conv_wavs_conv_2009_0003.wav", "english_EH1", "A"),
    (2009, "Blizzard_2009/B_submission_directory_english_EH1_2009_conv_wavs_conv_2009_0003.wav", "english_EH1", "B"),
    (2009, "Blizzard_2009/L_submission_directory_mandarin_MH_2009_news_wav_MH_2009_0001.wav", "mandarin_MH", "L"),

    # 2010: same as 2009.
    (2010, "Blizzard_2010/A_submission_directory_english_EH1_2010_news_wavs_news_2010_0010.wav", "english_EH1", "A"),
    (2010, "Blizzard_2010/M_submission_directory_mandarin_MH1_2010_news_wav_news_2010_0021.wav", "mandarin_MH1", "M"),

    # 2011: single-task (news).
    (2011, "Blizzard_2011/A_submission_directory_2011_news_wav_news_2011_0005.wav", "news", "A"),
    (2011, "Blizzard_2011/B_submission_directory_2011_news_wav_news_2011_0005.wav", "news", "B"),

    # 2012: single-task.
    (2012, "Blizzard_2012/A_submission_directory_2012_news_wav_news_2012_0002.wav", "news", "A"),

    # 2013: task-language encoding.
    (2013, "Blizzard_2013/A_submission_directory_2013_EH1-English_audiobook_sentences_wav_booksent_2013_0040.wav", "EH1", "A"),
    (2013, "Blizzard_2013/C_submission_directory_2013_EH1-English_audiobook_sentences_wav_booksent_2013_0040.wav", "EH1", "C"),

    # 2016: single-task (audiobook).
    (2016, "Blizzard_2016/A_submission_directory_2016_audiobook_wav_TwelfthNight_0004.wav", "audiobook", "A"),

    # 2019: single-task (celebrity).
    (2019, "Blizzard_2019/A_submission_directory_2019_celebrity_wav_celebrity_2019_101631.wav", "celebrity", "A"),

    # 2020: subtask is a PREFIX before system letter.
    (2020, "Blizzard_2020/MH1_A_submission_directory_news_wav_news_0002_0012.wav", "MH1", "A"),
    (2020, "Blizzard_2020/MH1_B_submission_directory_news_wav_news_0002_0012.wav", "MH1", "B"),

    # 2023: system first, then year-task.
    (2023, "Blizzard_2023/A_2023-FH1_submission_directory_FH1_MOS_wav_FH1_MOS_0073.wav", "FH1", "A"),
    (2023, "Blizzard_2023/A_2023-FH1_submission_directory_FH1_MOS_wav_FH1_MOS_0082.wav", "FH1", "A"),
]


@pytest.mark.parametrize("year,filepath,expected_subtask,expected_system", PARSE_CASES)
def test_parse_filepath(year: int, filepath: str, expected_subtask: str, expected_system: str) -> None:
    subtask, system = parse_filepath(year, filepath)
    assert subtask == expected_subtask, f"subtask mismatch for {filepath}"
    assert system == expected_system, f"system mismatch for {filepath}"
