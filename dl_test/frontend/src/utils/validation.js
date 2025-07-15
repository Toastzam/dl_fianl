/**
 * 허용된 파일 확장자 목록 (이미지)
 */
const ALLOWED_IMAGE_EXTENSIONS = ['jpg', 'jpeg', 'png', 'gif'];

/**
 * 최대 파일 크기 (10MB)
 */
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;

/**
 * 파일 이름의 확장자를 확인하여 허용된 이미지 파일 형식인지 검사합니다.
 * 'jpg/png/gif' 등이 허용됩니다.
 * @param {string} fileName - 검사할 파일의 이름
 * @returns {boolean} 허용된 확장자일 경우 true, 아닐 경우 false
 */
export const validateFileType = (fileName) => {
  if (!fileName) return false;
  const extension = fileName.split('.').pop()?.toLowerCase();
  return ALLOWED_IMAGE_EXTENSIONS.includes(extension);
};

/**
 * 파일 크기가 최대 허용 크기(10MB)를 초과하지 않는지 검사합니다.
 * @param {number} fileSizeInBytes - 검사할 파일의 크기 (bytes)
 * @returns {boolean} 파일 크기가 허용 범위 내에 있으면 true, 아닐 경우 false
 */
export const validateFileSize = (fileSizeInBytes) => {
  return fileSizeInBytes <= MAX_FILE_SIZE_BYTES;
}; 