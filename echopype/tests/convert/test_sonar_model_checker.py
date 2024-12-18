import glob
from echopype.convert import is_AD2CP, is_AZFP, is_AZFP6, is_EK60, is_EK80, is_ER60
import pytest

@pytest.fixture(scope="session") 
def test_is_EK80_ek80_files():
    """Test that EK80 files are identified as EK80."""
    # Collect all .raw files in the ek80 directory
    ek80_files = glob.glob("echopype/test_data/ek80/*.raw")
    
    # Check that each file in ek80 is identified as EK80
    for test_file_path in ek80_files:
        assert is_EK80(test_file_path, storage_options={}) == True
        
@pytest.fixture(scope="session") 
def test_is_EK80_non_ek80_files():
    """Test that non-EK80 files are not identified as EK80."""
    # Collect all .raw files in the ek60 directory (non-EK80 files)
    ek60_files = glob.glob("echopype/test_data/ek60/*.raw")
    
    # Check that each file in ek60 is not identified as EK80
    for test_file_path in ek60_files:
        assert is_EK80(test_file_path, storage_options={}) == False
        
@pytest.fixture(scope="session")         
def test_is_EK60_ek60_files():
    """Check that EK60 files are identified as EK60"""
    # Collect all .raw files in the ek60 directory
    ek60_files = glob.glob("echopype/test_data/ek60/from_echopy/*.raw")
    
    # Check that each file in ek60 is identified as EK60
    for test_file_path in ek60_files:
        assert is_EK60(test_file_path, storage_options={}) == True
        
@pytest.fixture(scope="session") 
def test_is_EK60_non_ek60_files():
    """Check that non-EK60 files are not identified as EK60"""
    # Collect all .raw files in the ek80 directory (non-EK60 files)
    ek80_files = glob.glob("echopype/test_data/ek80/*.raw")
    
    # Check that each file in ek80 is not identified as EK60
    for test_file_path in ek80_files:
        assert is_EK60(test_file_path, storage_options={}) == False
        
@pytest.fixture(scope="session")         
def test_is_ER60_er60_files():
    """Check that EK60 files are identified as EK60"""
    # Collect all .raw files in the ek60 directory
    ek60_files = glob.glob("echopype/test_data/ek60/from_echopy/*.raw")
    
    # Check that each file in ek60 is identified as EK60
    for test_file_path in ek60_files:
        assert is_ER60(test_file_path, storage_options={}) == True
        
@pytest.fixture(scope="session") 
def test_is_ER60_non_er60_files():
    """Check that non-EK60 files are not identified as EK60"""
    # Collect all .raw files in the ek80 directory (non-EK60 files)
    ek80_files = glob.glob("echopype/test_data/ek80/*.raw")
    
    # Check that each file in ek80 is not identified as EK60
    for test_file_path in ek80_files:
        assert is_ER60(test_file_path, storage_options={}) == False
        
@pytest.fixture(scope="session")         
def test_is_AZFP6_valid_files():
    """Test that .azfp files are identified as valid AZFP files."""
    # Collect all .azfp files in the test directory
    azfp_files = glob.glob("echopype/test_data/azfp6/*.azfp")
    
    # Check that each file in azfp is identified as valid AZFP
    for test_file_path in azfp_files:
        assert is_AZFP6(test_file_path) == True
        
@pytest.fixture(scope="session") 
def test_is_AZFP6_invalid_files():
    """Test that non-.azfp files are not identified as valid AZFP files."""
    # Collect all non-.azfp files in the test directory
    non_azfp_files = glob.glob("echopype/test_data/azfp/*")
    
    # Check that each file in non_azfp is not identified as valid AZFP
    for test_file_path in non_azfp_files:
        assert is_AZFP6(test_file_path) == False
        
@pytest.fixture(scope="session")         
def test_is_AZFP_valid_files():
    """Test that XML files with <InstrumentType string="AZFP"> are identified as valid AZFP files."""
    # Collect all valid XML files in the test directory
    valid_files = glob.glob("echopype/test_data/azfp/*.xml") + glob.glob("echopype/test_data/azfp/*.XML")
    
    for test_file_path in valid_files:
        assert is_AZFP(test_file_path) == True
        
@pytest.fixture(scope="session") 
def test_is_AZFP_invalid_files():
    """Test that XML files without <InstrumentType string="AZFP"> are not identified as valid AZFP files."""
    # Collect all invalid XML files in the test directory
    invalid_files = glob.glob("echopype/test_data/azfp6/*")
    
    for test_file_path in invalid_files:
        assert is_AZFP(test_file_path) == False
        
@pytest.fixture(scope="session") 
def test_is_AD2CP_valid_files():
    """Test that .ad2cp files are identified as valid AD2CP files."""
    # Collect all .ad2cp files in the test directory
    ad2cp_files = glob.glob("echopype/test_data/ad2cp/normal/*.ad2cp")
    
    # Check that each file in ad2cp is identified as valid AD2CP
    for test_file_path in ad2cp_files:
        assert is_AD2CP(test_file_path) == True
        
@pytest.fixture(scope="session") 
def test_is_AD2CP_invalid_files():
    """Test that non-.ad2cp files are not identified as valid AD2CP files."""
    # Collect all non-.ad2cp files in the test directory
    non_ad2cp_files = glob.glob("echopype/test_data/azfp6/*")
    
    # Check that each file in non_ad2cp is not identified as valid AD2CP
    for test_file_path in non_ad2cp_files:
        assert is_AD2CP(test_file_path) == False