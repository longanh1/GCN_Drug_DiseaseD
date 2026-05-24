import {
  Injectable, ConflictException, UnauthorizedException,
  NotFoundException, BadRequestException, ForbiddenException,
} from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcrypt';
import { User } from '../users/user.entity';
import {
  RegisterDto, LoginDto, UpdateProfileDto, ChangePasswordDto,
  ChangeRoleDto, AdminCreateUserDto, AdminResetPasswordDto,
} from './auth.dto';

@Injectable()
export class AuthService {
  constructor(
    @InjectRepository(User)
    private usersRepo: Repository<User>,
    private jwtService: JwtService,
  ) {}

  // ── Register ──────────────────────────────────────────────────────
  async register(dto: RegisterDto) {
    const existing = await this.usersRepo.findOne({
      where: [{ email: dto.email }, { username: dto.username }],
    });
    if (existing) {
      if (existing.email === dto.email) throw new ConflictException('Email đã được sử dụng');
      throw new ConflictException('Tên đăng nhập đã tồn tại');
    }
    const passwordHash = await bcrypt.hash(dto.password, 12);
    const user = this.usersRepo.create({
      email: dto.email, username: dto.username, passwordHash,
      fullName: dto.fullName || null, role: 'user', isActive: true,
    });
    await this.usersRepo.save(user);
    return this._buildTokenResponse(user);
  }

  // ── Login ─────────────────────────────────────────────────────────
  async login(dto: LoginDto) {
    const user = await this.usersRepo.findOne({
      where: [{ email: dto.identifier }, { username: dto.identifier }],
    });
    if (!user) throw new UnauthorizedException('Email/tên đăng nhập hoặc mật khẩu không đúng');
    if (!user.isActive) throw new UnauthorizedException('Tài khoản đã bị khóa. Vui lòng liên hệ admin');
    const valid = await bcrypt.compare(dto.password, user.passwordHash);
    if (!valid) throw new UnauthorizedException('Email/tên đăng nhập hoặc mật khẩu không đúng');
    return this._buildTokenResponse(user);
  }

  // ── Get Profile ───────────────────────────────────────────────────
  async getProfile(userId: string) {
    const user = await this.usersRepo.findOne({ where: { id: userId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    return this._safeUser(user);
  }

  // ── Update Profile ────────────────────────────────────────────────
  async updateProfile(userId: string, dto: UpdateProfileDto) {
    const user = await this.usersRepo.findOne({ where: { id: userId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    if (dto.fullName !== undefined) user.fullName = dto.fullName;
    if (dto.avatarUrl !== undefined) user.avatarUrl = dto.avatarUrl;
    await this.usersRepo.save(user);
    return this._safeUser(user);
  }

  // ── Change Password (self) ────────────────────────────────────────
  async changePassword(userId: string, dto: ChangePasswordDto) {
    const user = await this.usersRepo.findOne({ where: { id: userId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    const valid = await bcrypt.compare(dto.currentPassword, user.passwordHash);
    if (!valid) throw new BadRequestException('Mật khẩu hiện tại không đúng');
    user.passwordHash = await bcrypt.hash(dto.newPassword, 12);
    await this.usersRepo.save(user);
    return { message: 'Đổi mật khẩu thành công' };
  }

  // ── Get permissions for current user ─────────────────────────────
  getPermissions(role: string) {
    const base = {
      view_predictions:     true,
      view_history:         role !== 'viewer',
      run_prediction:       role !== 'viewer',
      save_history:         role !== 'viewer',
      generate_molecule:    role !== 'viewer',
      view_model_stages:    true,
      update_own_profile:   true,
      change_own_password:  true,
      // Admin-only
      manage_users:         role === 'admin',
      change_user_role:     role === 'admin',
      lock_unlock_user:     role === 'admin',
      delete_user:          role === 'admin',
      create_user:          role === 'admin',
      reset_user_password:  role === 'admin',
      view_all_history:     role === 'admin',
      view_system_stats:    role === 'admin',
    };
    return base;
  }

  // ════════════════════════════════════════════════════════════════════
  // ADMIN METHODS
  // ════════════════════════════════════════════════════════════════════

  // ── List all users ────────────────────────────────────────────────
  async listUsers(search?: string) {
    let users = await this.usersRepo.find({ order: { createdAt: 'DESC' } });
    if (search) {
      const q = search.toLowerCase();
      users = users.filter(u =>
        u.email.toLowerCase().includes(q) ||
        u.username.toLowerCase().includes(q) ||
        (u.fullName || '').toLowerCase().includes(q),
      );
    }
    return users.map(this._safeUser);
  }

  // ── Get single user (admin) ───────────────────────────────────────
  async getUserById(userId: string) {
    const user = await this.usersRepo.findOne({ where: { id: userId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    return this._safeUser(user);
  }

  // ── Toggle active ─────────────────────────────────────────────────
  async toggleActive(targetId: string, adminId: string) {
    if (targetId === adminId) throw new ForbiddenException('Không thể tự khóa chính mình');
    const user = await this.usersRepo.findOne({ where: { id: targetId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    user.isActive = !user.isActive;
    await this.usersRepo.save(user);
    return this._safeUser(user);
  }

  // ── Change role ───────────────────────────────────────────────────
  async changeRole(targetId: string, dto: ChangeRoleDto, adminId: string) {
    if (targetId === adminId) throw new ForbiddenException('Không thể đổi role của chính mình');
    const user = await this.usersRepo.findOne({ where: { id: targetId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    user.role = dto.role;
    await this.usersRepo.save(user);
    return this._safeUser(user);
  }

  // ── Admin create user ─────────────────────────────────────────────
  async adminCreateUser(dto: AdminCreateUserDto) {
    const existing = await this.usersRepo.findOne({
      where: [{ email: dto.email }, { username: dto.username }],
    });
    if (existing) {
      if (existing.email === dto.email) throw new ConflictException('Email đã được sử dụng');
      throw new ConflictException('Tên đăng nhập đã tồn tại');
    }
    const passwordHash = await bcrypt.hash(dto.password, 12);
    const user = this.usersRepo.create({
      email: dto.email, username: dto.username, passwordHash,
      fullName: dto.fullName || null,
      role: dto.role || 'user',
      isActive: dto.isActive !== undefined ? dto.isActive : true,
    });
    await this.usersRepo.save(user);
    return this._safeUser(user);
  }

  // ── Admin reset password ──────────────────────────────────────────
  async adminResetPassword(targetId: string, dto: AdminResetPasswordDto) {
    const user = await this.usersRepo.findOne({ where: { id: targetId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    user.passwordHash = await bcrypt.hash(dto.newPassword, 12);
    await this.usersRepo.save(user);
    return { message: `Đã reset mật khẩu cho @${user.username}` };
  }

  // ── Delete user ───────────────────────────────────────────────────
  async deleteUser(targetId: string, adminId: string) {
    if (targetId === adminId) throw new ForbiddenException('Không thể xóa chính mình');
    const user = await this.usersRepo.findOne({ where: { id: targetId } });
    if (!user) throw new NotFoundException('Người dùng không tồn tại');
    await this.usersRepo.remove(user);
    return { message: `Đã xóa tài khoản @${user.username}` };
  }

  // ── System stats (admin) ──────────────────────────────────────────
  async getSystemStats() {
    const all   = await this.usersRepo.count();
    const active  = await this.usersRepo.count({ where: { isActive: true } });
    const admins  = await this.usersRepo.count({ where: { role: 'admin' } });
    const viewers = await this.usersRepo.count({ where: { role: 'viewer' } });
    const users   = await this.usersRepo.count({ where: { role: 'user' } });
    return { total: all, active, inactive: all - active, admins, users, viewers };
  }

  // ── Private Helpers ───────────────────────────────────────────────
  private _buildTokenResponse(user: User) {
    const payload = { sub: user.id, email: user.email, username: user.username, role: user.role };
    const token = this.jwtService.sign(payload);
    return { access_token: token, token_type: 'Bearer', user: this._safeUser(user) };
  }

  private _safeUser(user: User) {
    return {
      id: user.id, email: user.email, username: user.username,
      fullName: user.fullName, role: user.role, isActive: user.isActive,
      avatarUrl: user.avatarUrl, createdAt: user.createdAt,
    };
  }
}


