import {
  Controller, Post, Get, Patch, Delete, Body, Param,
  UseGuards, Request, HttpCode, HttpStatus, Query,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import {
  RegisterDto, LoginDto, UpdateProfileDto, ChangePasswordDto,
  ChangeRoleDto, AdminCreateUserDto, AdminResetPasswordDto,
} from './auth.dto';
import { JwtAuthGuard } from './jwt-auth.guard';
import { RolesGuard } from './roles.guard';
import { Roles } from './roles.decorator';
import { Public } from './public.decorator';

// ── Apply JWT guard to all routes, Roles guard for role checks ─────────
@UseGuards(JwtAuthGuard, RolesGuard)
@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  // ════════════════════════════════════════════════════════════════════
  // PUBLIC
  // ════════════════════════════════════════════════════════════════════

  @Public()
  @Post('register')
  register(@Body() dto: RegisterDto) {
    return this.authService.register(dto);
  }

  @Public()
  @Post('login')
  @HttpCode(HttpStatus.OK)
  login(@Body() dto: LoginDto) {
    return this.authService.login(dto);
  }

  // ════════════════════════════════════════════════════════════════════
  // AUTHENTICATED (any logged-in user)
  // ════════════════════════════════════════════════════════════════════

  /** Returns profile + permissions for the current user */
  @Get('me')
  getMe(@Request() req) {
    return this.authService.getProfile(req.user.id);
  }

  /** Returns permission map based on role */
  @Get('me/permissions')
  getPermissions(@Request() req) {
    return this.authService.getPermissions(req.user.role);
  }

  @Patch('me')
  updateMe(@Request() req, @Body() dto: UpdateProfileDto) {
    return this.authService.updateProfile(req.user.id, dto);
  }

  @Post('me/change-password')
  @HttpCode(HttpStatus.OK)
  changePassword(@Request() req, @Body() dto: ChangePasswordDto) {
    return this.authService.changePassword(req.user.id, dto);
  }

  // ════════════════════════════════════════════════════════════════════
  // ADMIN ONLY
  // ════════════════════════════════════════════════════════════════════

  /** List users with optional search */
  @Roles('admin')
  @Get('admin/users')
  listUsers(@Query('search') search?: string) {
    return this.authService.listUsers(search);
  }

  /** System user statistics */
  @Roles('admin')
  @Get('admin/stats')
  systemStats() {
    return this.authService.getSystemStats();
  }

  /** Get single user detail */
  @Roles('admin')
  @Get('admin/users/:id')
  getUser(@Param('id') id: string) {
    return this.authService.getUserById(id);
  }

  /** Admin manually creates a user (any role) */
  @Roles('admin')
  @Post('admin/users')
  @HttpCode(HttpStatus.CREATED)
  createUser(@Body() dto: AdminCreateUserDto) {
    return this.authService.adminCreateUser(dto);
  }

  /** Lock / unlock account */
  @Roles('admin')
  @Patch('admin/users/:id/toggle')
  toggleUser(@Request() req, @Param('id') id: string) {
    return this.authService.toggleActive(id, req.user.id);
  }

  /** Change role: admin ↔ user ↔ viewer */
  @Roles('admin')
  @Patch('admin/users/:id/role')
  changeRole(@Request() req, @Param('id') id: string, @Body() dto: ChangeRoleDto) {
    return this.authService.changeRole(id, dto, req.user.id);
  }

  /** Admin resets another user's password */
  @Roles('admin')
  @Post('admin/users/:id/reset-password')
  @HttpCode(HttpStatus.OK)
  resetPassword(@Param('id') id: string, @Body() dto: AdminResetPasswordDto) {
    return this.authService.adminResetPassword(id, dto);
  }

  /** Permanently delete a user */
  @Roles('admin')
  @Delete('admin/users/:id')
  deleteUser(@Request() req, @Param('id') id: string) {
    return this.authService.deleteUser(id, req.user.id);
  }
}

