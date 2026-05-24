import { IsEmail, IsString, MinLength, MaxLength, Matches, IsOptional, IsIn, IsBoolean } from 'class-validator';

export class RegisterDto {
  @IsEmail({}, { message: 'Email không hợp lệ' })
  email: string;

  @IsString()
  @MinLength(3, { message: 'Tên đăng nhập ít nhất 3 ký tự' })
  @MaxLength(50)
  @Matches(/^[a-zA-Z0-9_]+$/, { message: 'Tên đăng nhập chỉ gồm chữ, số, dấu _' })
  username: string;

  @IsString()
  @MinLength(8, { message: 'Mật khẩu ít nhất 8 ký tự' })
  @MaxLength(100)
  password: string;

  @IsOptional()
  @IsString()
  @MaxLength(120)
  fullName?: string;
}

export class LoginDto {
  @IsString()
  identifier: string; // email or username

  @IsString()
  password: string;
}

export class UpdateProfileDto {
  @IsOptional()
  @IsString()
  @MaxLength(120)
  fullName?: string;

  @IsOptional()
  @IsString()
  avatarUrl?: string;
}

export class ChangePasswordDto {
  @IsString()
  currentPassword: string;

  @IsString()
  @MinLength(8)
  newPassword: string;
}

// ── Admin DTOs ─────────────────────────────────────────────────────────────
export class ChangeRoleDto {
  @IsString()
  @IsIn(['admin', 'user', 'viewer'], { message: 'Role phải là admin, user hoặc viewer' })
  role: 'admin' | 'user' | 'viewer';
}

export class AdminCreateUserDto {
  @IsEmail({}, { message: 'Email không hợp lệ' })
  email: string;

  @IsString()
  @MinLength(3)
  @MaxLength(50)
  @Matches(/^[a-zA-Z0-9_]+$/, { message: 'Tên đăng nhập chỉ gồm chữ, số, dấu _' })
  username: string;

  @IsString()
  @MinLength(8)
  @MaxLength(100)
  password: string;

  @IsOptional()
  @IsString()
  @MaxLength(120)
  fullName?: string;

  @IsOptional()
  @IsString()
  @IsIn(['admin', 'user', 'viewer'])
  role?: 'admin' | 'user' | 'viewer';

  @IsOptional()
  @IsBoolean()
  isActive?: boolean;
}

export class AdminResetPasswordDto {
  @IsString()
  @MinLength(8)
  newPassword: string;
}
